#include <cuda_runtime_api.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#define GLCPP_DEFAULT(x) // avoid dynamic initialization in CUDA constant vars
#ifdef INCLUDE_BASE_WAVEFRONT_NEURAL_PT
#include "wavefront_neural_pt.h"
#endif

#include "neural-lod/neural_lod_learning.h"
#include "neural-lod/neural_lod_learning_common.cuh"
#include "neural-lod/neural_lod_learning_grid.cuh"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>

// Framebuffer
namespace clsl {

__constant__ cudaSurfaceObject_t framebuffer;
__constant__ cudaSurfaceObject_t accum_buffer;
__constant__ int fb_width;
__constant__ int fb_height;

__constant__ NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb_const;
__constant__ float voxel_size_const;
__constant__ int current_lod_const;
__constant__ float voxel_extent_dilation_outward_const;
__constant__ float max_cam_to_aabox_length_const;
__constant__ float max_visibility_rr_prob_const;

} // namespace

// Scene & Includes
namespace clsl {
    using namespace glm;
    #define inline __device__ inline

    #include "../rendering/language.hpp"
    #include "../rendering/pointsets/lcg_rng.glsl"
    #include "../rendering/util.glsl"
    
    #include "../vulkan/gpu_params.glsl"

    __constant__ ViewParams view_params;
    __constant__ RenderParams render_params;
    __constant__ SceneParams scene_params;

    #undef inline
} // namespace

// Neural Path Tracing
namespace clsl {

inline __device__ bool rr_terminate(uint32_t bounce, uint32_t rr_start_bounce, vec3& path_throughput, RANDOM_STATE& rng) {
    // Russian roulette termination
    if (bounce >= rr_start_bounce) {
        // Since our path throughput is an estimate itself it's important we clamp it to 1
        // aswell
        float prefix_weight = min(1.0, max(path_throughput.x, max(path_throughput.y, path_throughput.z)));

        float rr_prob = prefix_weight;
        float rr_sample = RANDOM_FLOAT1(rng, -1);
        if (rr_sample < rr_prob)
            path_throughput /= rr_prob;
        else
            return false;
    }

    return true;
}

inline __device__ bool visibility_rr_terminate(float inference_value, uint32_t visibility_inference, vec3& path_throughput, RANDOM_STATE& rng) {
    // Russian roulette termination
    if (visibility_inference >= 5) {
        float rr_prob =  max(inference_value, max_visibility_rr_prob_const);
        float rr_sample = RANDOM_FLOAT1(rng, -1);
        if (rr_sample < rr_prob)
            path_throughput /= rr_prob;
        else
            return false;
    }

    return true;
}

inline __device__ void raymarch_impl(int i,
                                     NeuralBounceQuery *bounce_queries,
                                     NeuralBounceResult *bounce_results,
                                     uint32_t bounce,
                                     uint32_t visibility_inference,
                                     const nanovdb::FloatGrid *d_grid)
{
    // Change to float for slight speed improvement (can introduce floating point errors) 
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    NeuralBounceQuery &neural_query = bounce_queries[i];
    NeuralBounceResult &neural_result = bounce_results[i];
    
    if(neural_query.path_state == ACTIVE_PATH_WITH_INTERSECTION || neural_query.path_state == INACTIVE_PATH)
        return;

    assert(neural_query.path_state == ACTIVE_PATH);

    vec3 ray_origin = neural_query.origin;
    vec3 ray_dir = neural_query.dir;

    RayT index_ray(vec3_to_nanovdb<RealT>(ray_origin), vec3_to_nanovdb<RealT>(ray_dir), 0.0);
    if (neural_result.t_voxel_near != -1.f &&
        neural_result.t_voxel_far != -1.f) {
        // unocluded and at the same lod level as the previous iteration
        index_ray.setMinTime(
            (neural_result.t_voxel_near + neural_result.t_voxel_far) * 0.5);
    }

    nanovdb::Coord last_lod_voxel_idx = ivec3_to_coord(neural_result.lod_voxel_idx);
    nanovdb::Coord lod_voxel_idx;
    RealT t_index_ray_start = 0.f;
    RealT t_index_ray_end = 0.f;
    float voxel_threshold;
    auto acc = d_grid->tree().getAccessor();
    if (first_active_voxel_intersection(index_ray,
                                        acc,
                                        lod_voxel_idx,
                                        t_index_ray_start,
                                        t_index_ray_end,
                                        voxel_threshold,
                                        last_lod_voxel_idx)) {
        neural_result.lod_voxel_idx = coord_to_ivec3(lod_voxel_idx);
        neural_result.t_voxel_near = t_index_ray_start;
        neural_result.t_voxel_far = t_index_ray_end;
        neural_result.voxel_threshold = voxel_threshold;
    } else {
        neural_query.path_state = ESCAPED_PATH;
    }
    return;
}

inline __device__ void generate_visibility_inference_data_impl(uint32_t i,
                                                   NeuralBounceQuery *bounce_queries,
                                                   NeuralBounceResult *bounce_results,
                                                   float *__restrict__ visibility_input_data,
                                                   float inference_min_extent_dilation)
{
    NeuralBounceQuery &neural_query = bounce_queries[i];
    NeuralBounceResult &neural_result = bounce_results[i];

    if (neural_query.path_state != ACTIVE_PATH) {
        return;
    }

    vec3 ray_dir = neural_query.dir;
    vec3 wi = -ray_dir;

    neural_lod_learning::NeuralLodLearning::Voxel voxel =
        compute_lod_voxel(neural_result.lod_voxel_idx,
                          clsl::voxel_grid_aabb_const,
                          clsl::voxel_size_const,
                          current_lod_const);

    // first map to [-0.5,0.5]
    const bool clamp_to_range = true;
    vec3 voxel_near_pos = neural_query.origin + neural_query.dir * neural_result.t_voxel_near;
    vec3 voxel_far_pos = neural_query.origin + neural_query.dir * neural_result.t_voxel_far;
    vec3 local_pi =
        index_to_local_voxel_pos(voxel_near_pos, neural_result.lod_voxel_idx, clamp_to_range);
    vec3 local_po =
        index_to_local_voxel_pos(voxel_far_pos, neural_result.lod_voxel_idx, clamp_to_range);

    vec3 local_pi_po[2] = {local_pi,local_po};
    get_deterministic_local_pi_po(local_pi_po,local_pi,local_po);

    const float min_extent_dilation =
        min(inference_min_extent_dilation, voxel_extent_dilation_outward_const);
    local_pi *= (1.f + min_extent_dilation);
    local_po *= (1.f + min_extent_dilation);

    vec3 pi = voxel.center + local_pi * voxel.extent;
    vec3 po = voxel.center + local_po * voxel.extent;
    vec3 input_pi = (pi - voxel_grid_aabb_const.min) / voxel_grid_aabb_const.extent();
    vec3 input_po = (po - voxel_grid_aabb_const.min) / voxel_grid_aabb_const.extent();

    uint32_t visibility_input_idx = i * neural_lod_learning::visibility_n_input_dims;

    visibility_input_data[visibility_input_idx + 0] = input_pi.x;
    visibility_input_data[visibility_input_idx + 1] = input_pi.y;
    visibility_input_data[visibility_input_idx + 2] = input_pi.z;

    visibility_input_data[visibility_input_idx + 3] = input_po.x;
    visibility_input_data[visibility_input_idx + 4] = input_po.y;
    visibility_input_data[visibility_input_idx + 5] = input_po.z;

}

inline __device__ void generate_throughput_inference_data_impl(uint32_t i,
                                                   NeuralBounceQuery *bounce_queries,
                                                   NeuralBounceResult *bounce_results,
                                                   float *__restrict__ throughput_input_data,
                                                   bool use_hg,
                                                   float hg_g)
{
    NeuralBounceQuery &neural_query = bounce_queries[i];
    NeuralBounceResult &neural_result = bounce_results[i];

    // If not we are doing useless work
    assert(neural_query.path_state == ACTIVE_PATH_WITH_INTERSECTION);

    vec3 ray_dir = neural_query.dir;
    vec3 wo = -ray_dir;

    vec2 dir_sample = RANDOM_FLOAT2(neural_query.rng, -1);
    vec3 wi = use_hg ? square_to_hg(dir_sample, ray_dir, hg_g)
                     : square_to_uniform_sphere(dir_sample);

    neural_lod_learning::NeuralLodLearning::Voxel voxel =
        compute_lod_voxel(neural_result.lod_voxel_idx,
                          clsl::voxel_grid_aabb_const,
                          clsl::voxel_size_const,
                          clsl::current_lod_const);

    // Normalized center coordinates
    vec3 input_pos = (voxel.center - clsl::voxel_grid_aabb_const.min) /
                     clsl::voxel_grid_aabb_const.extent();

    uint32_t throughput_input_idx = i * neural_lod_learning::throughput_n_input_dims;

    throughput_input_data[throughput_input_idx + 0] = input_pos.x;
    throughput_input_data[throughput_input_idx + 1] = input_pos.y;
    throughput_input_data[throughput_input_idx + 2] = input_pos.z;

    vec3 input_wo = (wo + 1.f) * 0.5f;
    throughput_input_data[throughput_input_idx + 3] = input_wo.x;
    throughput_input_data[throughput_input_idx + 4] = input_wo.y;
    throughput_input_data[throughput_input_idx + 5] = input_wo.z;

    vec3 input_wi = (wi + 1.f) * 0.5f;
    throughput_input_data[throughput_input_idx + 6] = input_wi.x;
    throughput_input_data[throughput_input_idx + 7] = input_wi.y;
    throughput_input_data[throughput_input_idx + 8] = input_wi.z;

    neural_result.throughput_wi = wi;
    if(use_hg){
        float pdf = square_to_hg_pdf(dot(ray_dir, wi), hg_g);
        assert(pdf != 0.0f);
        neural_result.throughput_inference_pdf_inv = 1.f / pdf;
    }else{
        neural_result.throughput_inference_pdf_inv = 4.f * M_PI;
    }
    
}

__device__ inline glm::vec3 get_throughput_inference(
    int i, float *__restrict__ throughput_output_data, bool log_learning)
{
    uint32_t output_idx = i * neural_lod_learning::throughput_n_output_dims;
    vec3 throughput_inference;
    throughput_inference.x = throughput_output_data[output_idx + 0];
    throughput_inference.y = throughput_output_data[output_idx + 1];
    throughput_inference.z = throughput_output_data[output_idx + 2];

    if(log_learning){
        throughput_inference.x = expf(throughput_inference.x) - 1.f;
        throughput_inference.y = expf(throughput_inference.y) - 1.f;
        throughput_inference.z = expf(throughput_inference.z) - 1.f;
    }
    throughput_inference = max(vec3(0.f), throughput_inference);
    
    return throughput_inference;
}

__device__ inline void accumulate_neural_bounce(int sample_index,
                                                const NeuralBounceQuery &neural_query,
                                                const NeuralBounceResult &neural_result)
{
    assert(neural_query.path_state == INACTIVE_PATH);

    ivec2 pixel = coord_1d_to_2d(neural_result.pixel_coord,fb_width,fb_height);
    vec3 illum = neural_result.illumination;

    // Computing variance
    if (render_params.output_moment == 1){
#ifdef STATS_NUM_BOUNCES
        vec3 accum_bounce = vec3(illum.x, illum.y, illum.z);
        float4 accum_data = { accum_bounce.x, accum_bounce.y, accum_bounce.z, 1.0f };
        surf2Dwrite(accum_data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
        return;
#endif
#ifdef STATS_VARIANCE
        vec3 accum_moment = illum*illum;
        float4 accum_data = { accum_moment.x, accum_moment.y, accum_moment.z, 1.0f };
        surf2Dwrite(accum_data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
        return;
#endif
    }

    vec3 accum_color = illum;
    {
        float4 data = {accum_color.x, accum_color.y, accum_color.z, 1.0f };
        surf2Dwrite(
            data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
    }
}

} // namespace


#ifdef INCLUDE_BASE_WAVEFRONT_NEURAL_PT
__forceinline__ void WavefrontNeuralPT::this_module_setup_constants(
    neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
    float voxel_size,
    int current_lod,
    float voxel_extent_dilation_outward,
    float max_cam_to_aabox_length)
{
    checkCuda(cudaMemcpyToSymbol(clsl::framebuffer, &backend->pixel_buffer(), sizeof(clsl::framebuffer)));
    checkCuda(cudaMemcpyToSymbol(clsl::accum_buffer, &backend->accum_buffer(), sizeof(clsl::accum_buffer)));
    checkCuda(cudaMemcpyToSymbol(clsl::fb_width, &backend->screen_width, sizeof(clsl::fb_width)));
    checkCuda(cudaMemcpyToSymbol(clsl::fb_height, &backend->screen_height, sizeof(clsl::fb_height)));

    auto global_params_buffer = (clsl::GlobalParams*) backend->global_params_buffer;
    auto local_host_params = (clsl::LocalParams*) backend->local_host_params;

    checkCuda(cudaMemcpyToSymbol(clsl::view_params, &local_host_params->view_params, sizeof(clsl::view_params), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(clsl::scene_params, &global_params_buffer->scene_params, sizeof(clsl::scene_params), 0, cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpyToSymbol(clsl::render_params, &global_params_buffer->render_params, sizeof(clsl::render_params), 0, cudaMemcpyDeviceToDevice));

    checkCuda(cudaMemcpyToSymbol(clsl::view_params, &accumulated_spp, sizeof(accumulated_spp), offsetof(clsl::ViewParams, frame_id)));

    // Voxel grid parameters
    checkCuda(cudaMemcpyToSymbol(
        clsl::voxel_grid_aabb_const, &voxel_grid_aabb, sizeof(clsl::voxel_grid_aabb_const)));
    checkCuda(cudaMemcpyToSymbol(
        clsl::voxel_size_const, &voxel_size, sizeof(clsl::voxel_size_const)));
    checkCuda(cudaMemcpyToSymbol(
        clsl::current_lod_const, &current_lod, sizeof(clsl::current_lod_const)));
    checkCuda(cudaMemcpyToSymbol(clsl::voxel_extent_dilation_outward_const,
                                 &voxel_extent_dilation_outward,
                                 sizeof(clsl::voxel_extent_dilation_outward_const)));
    checkCuda(cudaMemcpyToSymbol(clsl::max_cam_to_aabox_length_const,
                                 &max_cam_to_aabox_length,
                                 sizeof(clsl::max_cam_to_aabox_length_const)));

    float visibility_rr_prob = std::pow(1e-6f,1.f/ (3*(VOXEL_GRID_MAX_RES())));

    checkCuda(cudaMemcpyToSymbol(clsl::max_visibility_rr_prob_const, &visibility_rr_prob, sizeof(float)));

    checkCuda(cudaDeviceSynchronize());
}
#endif
