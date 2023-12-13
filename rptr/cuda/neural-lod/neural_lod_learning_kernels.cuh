#include <tiny-cuda-nn/common.h>

#include "neural_lod_learning.h"
#include "neural_lod_learning_common.cuh"
#include "neural_lod_learning_grid.cuh"

using namespace glm;
#define inline __host__ __device__ inline
namespace {
#include "../../rendering/language.hpp"
#include "../../rendering/pointsets/lcg_rng.glsl"
}
#undef inline

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN

__constant__ cudaSurfaceObject_t framebuffer;
__constant__ cudaTextureObject_t accum_buffer;
__constant__ cudaTextureObject_t debug_buffer;

__constant__ int fb_width;
__constant__ int fb_height;
__constant__ bool depth_truncation_const;

__constant__ NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb_const;
__constant__ float voxel_size_const;
__constant__ int current_lod_const;

inline __device__ double exp_pdf(int lod, uint32_t init_pow_2){
    return scalbn(1.0, -(init_pow_2 * (lod+1)));
}

// s : slope strength [0,inf)
// d : density shift towards lower or higher lod [-0.5,0.5]
inline __device__ double shifted_square_pdf(int lod, float s, float d){
    const float x = (float(lod)/NUM_LODS() + d - 0.5f + 0.0625f);
    return 1.0 + s * x * x;
}

inline __device__ uint32_t sample_discrete_lods(float sample, float strength, float shift)
{
    uint32_t selected_lod;
// #define UNIFORM_LOD_SAMPLING
#ifdef UNIFORM_LOD_SAMPLING
    selected_lod = (uint32_t)(sample * NUM_LODS()) % NUM_LODS();
#else
    // Compute the cdf for each value in the set {0, 1, 2, 3, 4, 5, 6, 7}
    double cdf[NUM_LODS()];
    // uint32_t init_pow_2 = 1; // 1 = 1/2 , 2 = 1/4, 3 = 1/8
    for (size_t lod = 0; lod < NUM_LODS(); lod++) {
        // float pdf = exp_pdf(lod,init_pow_2);
        float pdf = shifted_square_pdf(lod,strength,shift);
        cdf[lod] = lod == 0 ? pdf : cdf[lod-1] + pdf;
    }

    #pragma unroll
    for (size_t i = 0; i < NUM_LODS(); i++) {
        cdf[i] /= cdf[NUM_LODS()-1];
    }   
   
    for (size_t i = 0; i < NUM_LODS(); i++)
    {
        if(sample < cdf[i] ){
            selected_lod = i;
            break;
        }
    }
#endif
    return selected_lod;
}

inline __device__ NeuralLodLearning::Voxel sample_voxel(
    LCGRand &lcg,
    uint32_t *morton_sparse_voxel_idx_mapping,
    uint32_t *inferences_sample_count_grid,
    uint32_t *inference_total_sample_count,
    int sparse_voxel_count,
    uint32_t &selected_lod,
    float pdf_strength,
    float pdf_shift)
{
    uint32_t sparse_idx =
        (uint32_t)(lcg_randomf(lcg) * sparse_voxel_count) % sparse_voxel_count;
    assert(sparse_idx < sparse_voxel_count);
    uint32_t morton_idx = morton_sparse_voxel_idx_mapping[sparse_idx];

    float sample = lcg_randomf(lcg);
    uint32_t lod_level =
        sample_discrete_lods(sample, pdf_strength, pdf_shift);
    glm::uvec3 lod_xyz = decode_morton3(morton_idx);
    lod_xyz.x >>= lod_level;
    lod_xyz.y >>= lod_level;
    lod_xyz.z >>= lod_level;
    selected_lod = lod_level;
 
    return compute_lod_voxel(
        lod_xyz, voxel_grid_aabb_const, voxel_size_const, lod_level);
}

__global__ void fill_ray_queries_throughput(int batch_size,
                                            int sparse_voxel_count,
                                            uint32_t *morton_sparse_voxel_idx_mapping,
                                            int batch_index,
                                            ivec3 voxel_index,
                                            RenderRayQuery *queries,
                                            float *__restrict__ input_data,
                                            uint32_t *inferences_sample_count_grid,
                                            uint32_t *inference_total_sample_count,
                                            float voxel_extent_dilation_outward,
                                            float voxel_extent_dilation_inward,
                                            float voxel_extent_min_dilation,
                                            float voxel_bound_bias,
                                            int max_throughput_orders,
                                            int training_step,
                                            float pdf_strength,
                                            float pdf_shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(i < batch_size))
        return;
    
    uint32_t input_idx = i * throughput_n_input_dims;

    auto lcg = get_lcg_rng(batch_index*batch_size, 0, i);
    NeuralLodLearning::Voxel voxel;
    uint32_t selected_lod;
    if (voxel_index != ivec3(-1)){
        voxel = compute_lod_voxel(voxel_index,voxel_grid_aabb_const, voxel_size_const, current_lod_const);
        selected_lod = current_lod_const;
    }else{
        voxel = sample_voxel(lcg,
                             morton_sparse_voxel_idx_mapping,
                             inferences_sample_count_grid,
                             inference_total_sample_count,
                             sparse_voxel_count,
                             selected_lod,
                             pdf_strength,
                             pdf_shift);
    }

    float sampling_dist = 2.f * (1.f + voxel_extent_dilation_outward);
    float sampled_voxel_extent_dilation;
 
    vec3 wo = square_to_uniform_sphere(vec2(lcg_randomf(lcg),lcg_randomf(lcg)));
    vec3 ray_dir = -wo;
    vec3 local_po;

    float min_abs_cmp = min_component(abs(ray_dir));
    vec3 wo_perp_u(0.f);
    if(abs(ray_dir[0]) == min_abs_cmp){
        wo_perp_u.y = wo.z;
        wo_perp_u.z = -wo.y; 
                wo_perp_u.z = -wo.y; 
        wo_perp_u.z = -wo.y; 
                wo_perp_u.z = -wo.y; 
        wo_perp_u.z = -wo.y; 
    }else if (abs(ray_dir[1]) == min_abs_cmp){
        wo_perp_u.x = -wo.z;
        wo_perp_u.z = wo.x; 
    }else{
        wo_perp_u.x = -wo.y;
        wo_perp_u.y = wo.x; 
    }
    assert(abs(dot(wo_perp_u,wo)) <= 1e-5);
    vec3 wo_perp_v = cross(wo, wo_perp_u);

    vec3 ortho_cam_origin = wo * sampling_dist;

    // Randomly select one of the cube we will intersect BEFORE we perform the
    // rejection sampling
    // we randomly select inward or outward dilation depending on their ratio
    if (lcg_randomf(lcg) < voxel_extent_dilation_outward / (voxel_extent_dilation_inward +
                                                            voxel_extent_dilation_outward)) {
        // sample outward
        sampled_voxel_extent_dilation =
            lcg_randomf(lcg) * (voxel_extent_dilation_outward - voxel_extent_min_dilation) +
            voxel_extent_min_dilation;  // in [min_dilation_eps; +outward_dilation]
    } else {
        // sample inward
        sampled_voxel_extent_dilation =
            -(lcg_randomf(lcg) * (voxel_extent_dilation_inward - voxel_extent_min_dilation) +
              voxel_extent_min_dilation);  // in [-inward_dilation, -min_dilation_eps]
    }

    // Perform rejection sampling of the projected area;
    // sampling_dist has to be at least larger than the diagonal of the cube (sqrt(3))
    while (true) {
        // offset by maximum the voxel's diagonal on the projection plane at sqrt(3.0)
        // distance
        vec2 offset = (vec2(lcg_randomf(lcg), lcg_randomf(lcg)) - 0.5f) *
                        sampling_dist;  // > [-0.5 * sqrt(3)/2 ; + 0.5 * sqrt(3.0)/2]
        vec3 visibility_ray_origin =
            ortho_cam_origin + wo_perp_u * offset.x + wo_perp_v * offset.y;

        // Intersection with random aabb, any in [-0.5*(1-eps),0.5*(1 + eps)]
        float t_near, t_far;
        if (intersect_aabb(visibility_ray_origin,
                            ray_dir,
                            glm::vec3(0.f),
                            1.0f + sampled_voxel_extent_dilation,
                            t_near,
                            t_far)) {
            assert(t_near != -1e20 && t_far != 1e20f);
            local_po = visibility_ray_origin + ray_dir * t_near;
            break;
        }
    }

    // assert(abs(max_component(abs(local_po)) - 0.5f) < 1e-5);

    vec3 ray_origin = voxel.center + local_po * voxel.extent; 

    queries[i].origin = ray_origin;
    queries[i].voxel_center = voxel.center;
    queries[i].voxel_extent = voxel.extent * (1.f + max(sampled_voxel_extent_dilation,voxel_bound_bias));
    queries[i].dir = ray_dir;

    queries[i].t_max = 1e20f;
    queries[i].mode_or_data = max_throughput_orders;

    if (input_data != nullptr){

        // Normalize input
        vec3 input_pos = (voxel.center - voxel_grid_aabb_const.min) / voxel_grid_aabb_const.extent();

        input_data[input_idx + 0] = input_pos.x;
        input_data[input_idx + 1] = input_pos.y;
        input_data[input_idx + 2] = input_pos.z;

        // Setting normalized wo
        vec3 input_wo = (wo + 1.0f) * 0.5f;
        input_data[input_idx + 3] = input_wo.x;
        input_data[input_idx + 4] = input_wo.y;
        input_data[input_idx + 5] = input_wo.z;

    }
}

__global__ void record_ray_query_results_throughput_optim(int batch_size,
                                                      RenderRayQueryResult const *ray_results,
                                                      float *__restrict__ result,
                                                      float *__restrict__ input_data,
                                                      int * invalid_samples,
                                                      bool log_learning)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    uint32_t input_idx = i * throughput_n_input_dims;
    uint32_t output_idx = i * throughput_n_output_dims;

    const RenderRayQueryResult &ray_result = ray_results[i];

    if(log_learning){
        result[output_idx + 0] =  logf(1.f + ray_result.result.r);
        result[output_idx + 1] =  logf(1.f + ray_result.result.g);
        result[output_idx + 2] =  logf(1.f + ray_result.result.b);
    }else{
        result[output_idx + 0] = ray_result.result.r;
        result[output_idx + 1] = ray_result.result.g;
        result[output_idx + 2] = ray_result.result.b;
    }

    // Setting normalized wi
    vec3 input_wi = (ray_result.wi + 1.0f) * 0.5f;
    input_data[input_idx + 6] = input_wi.x;
    input_data[input_idx + 7] = input_wi.y;
    input_data[input_idx + 8] = input_wi.z;

    if (ray_result.occlusion_hit_or_pdf != 1) {
        atomicAdd(invalid_samples,1);
        for (uint32_t n = 0; n < throughput_n_output_dims; n++)
        {
            result[output_idx + n] = -1.f;
        }

        for (uint32_t n = 0; n < throughput_n_input_dims; n++)
        {
            input_data[input_idx + n] = -1.f;
        }
    }
}

__global__ void copy_valid_results_throughput_optim(int batch_size, 
                                                int* valid_samples,
                                                float *__restrict__ result,
                                                float *__restrict__ input_data,
                                                float *__restrict__ valid_result,
                                                float *__restrict__ valid_input_data,
                                                int max_valid_batch_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    uint32_t input_idx = i * throughput_n_input_dims;
    uint32_t output_idx = i * throughput_n_output_dims;

    // Don't copy invalid results
    if(result[output_idx + 0] == -1.f){
        return;
    }

    int valid_i = atomicAdd(valid_samples, 1);

    // Revert the last valid sample if larger than the maximum 
    // number of valid samples allowed
    if (valid_i >= max_valid_batch_samples){
        atomicAdd(valid_samples, -1);
        return;
    }

    uint32_t valid_input_idx = valid_i * throughput_n_input_dims;
    uint32_t valid_output_idx = valid_i * throughput_n_output_dims;

    for (uint32_t n = 0; n < throughput_n_output_dims; n++)
    {
        valid_result[valid_output_idx + n] = result[output_idx + n];
    }

    for (uint32_t n = 0; n < throughput_n_input_dims; n++)
    {
        valid_input_data[valid_input_idx + n] = input_data[input_idx + n];
    }
}

__global__ void record_reference_throughput_estimates(int batch_size,
                                                RenderRayQuery const *ray_queries,
                                                RenderRayQueryResult const *ray_results,
                                                float *ref_output_accumulated,
                                                uvec2 sampling_res,
                                                bool log_learning)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    const RenderRayQuery &ray_query = ray_queries[i];
    const RenderRayQueryResult &ray_result = ray_results[i];

    if (ray_result.occlusion_hit_or_pdf != 1) {
        return;
    }else{
        uint32_t output_idx = compute_inference_idx_for_throughput(-ray_query.dir,ray_result.wi, sampling_res, (throughput_n_output_dims + 1));

        if (log_learning) {
            atomicAdd(&ref_output_accumulated[output_idx + 0], logf(1.f + ray_result.result.r));
            atomicAdd(&ref_output_accumulated[output_idx + 1], logf(1.f + ray_result.result.g));
            atomicAdd(&ref_output_accumulated[output_idx + 2], logf(1.f + ray_result.result.b));
        } else {
            atomicAdd(&ref_output_accumulated[output_idx + 0], ray_result.result.r);
            atomicAdd(&ref_output_accumulated[output_idx + 1], ray_result.result.g);
            atomicAdd(&ref_output_accumulated[output_idx + 2], ray_result.result.b);
        }
        atomicAdd(&ref_output_accumulated[output_idx + 3], 1.0);
    }
}

__global__ void fill_ray_queries_visibility(int batch_size,
                                        int sparse_voxel_count,
                                        uint32_t* morton_sparse_voxel_idx_mapping,
                                        int batch_index,
                                        ivec3 voxel_index,
                                        RenderRayQuery *queries,
                                        float *__restrict__ input_data,
                                        uint32_t* inferences_sample_count_grid,
                                        uint32_t *inference_total_sample_count,
                                        float voxel_extent_dilation_outward,
                                        float voxel_extent_dilation_inward,
                                        float voxel_extent_min_dilation,
                                        float voxel_bound_bias,
                                        bool increase_boundary_density,
                                        bool fill_for_visualisation,
                                        float pdf_strength,
                                        float pdf_shift,
                                        int threshold_optim_lod)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(i < batch_size))
        return;
    
    uint32_t input_idx = i * visibility_n_input_dims;

    auto lcg = get_lcg_rng(batch_index*batch_size, 0, i);

    NeuralLodLearning::Voxel voxel;
    uint32_t selected_lod;
    if (voxel_index != ivec3(-1)){
        if(threshold_optim_lod != -1){
            uint32_t num_processed_voxels = sparse_voxel_count;
            uint32_t num_samples = batch_index;

            if (i >= num_samples * num_processed_voxels)
                return;

            ivec2 sample_idx_voxel_offset = coord_1d_to_2d(i, num_samples, num_processed_voxels);
            int voxel_offset = sample_idx_voxel_offset.y;

            lcg = get_lcg_rng(voxel_index.x + voxel_offset, 0, sample_idx_voxel_offset.x);
  
            // voxel_index store the first index in the sparse structure
            assert(voxel_index.x == voxel_index.y && voxel_index.y == voxel_index.z);
            glm::uvec3 lod_xyz = decode_morton3(morton_sparse_voxel_idx_mapping[voxel_index.x + voxel_offset]);
            selected_lod = threshold_optim_lod;
            voxel = compute_lod_voxel(
                lod_xyz, voxel_grid_aabb_const, voxel_size_const, threshold_optim_lod);
        }else{
            voxel = compute_lod_voxel(voxel_index,voxel_grid_aabb_const, voxel_size_const, current_lod_const);
        }
    }else if(threshold_optim_lod != -1){
        // i also corresponds to the number of sparse voxels for threshold optimisation
        // /!\ we also use the given lod's sparse voxel mapping when using the threshold optim 
        glm::uvec3 lod_xyz = decode_morton3(morton_sparse_voxel_idx_mapping[i]);
        selected_lod = threshold_optim_lod;
        voxel = compute_lod_voxel(
            lod_xyz, voxel_grid_aabb_const, voxel_size_const, threshold_optim_lod);
    }else{
        voxel = sample_voxel(lcg,
                             morton_sparse_voxel_idx_mapping,
                             inferences_sample_count_grid,
                             inference_total_sample_count,
                             sparse_voxel_count,
                             selected_lod,
                             pdf_strength,
                             pdf_shift);
    }

    float sampling_dist = 2.f * (1.f + voxel_extent_dilation_outward);
    float sampled_voxel_extent_dilation;
    vec3 local_pi_po[2];
    vec3 pi_po[2];
 
    vec3 wi = square_to_uniform_sphere(vec2(lcg_randomf(lcg),lcg_randomf(lcg)));
    vec3 ortho_ray_dir = -wi;

    float min_abs_cmp = min_component(abs(ortho_ray_dir));
    vec3 wi_perp_u(0.f);
    if(abs(ortho_ray_dir[0]) == min_abs_cmp){
        wi_perp_u.y = wi.z;
        wi_perp_u.z = -wi.y; 
                wi_perp_u.z = -wi.y; 
        wi_perp_u.z = -wi.y; 
                wi_perp_u.z = -wi.y; 
        wi_perp_u.z = -wi.y; 
    }else if (abs(ortho_ray_dir[1]) == min_abs_cmp){
        wi_perp_u.x = -wi.z;
        wi_perp_u.z = wi.x; 
    }else{
        wi_perp_u.x = -wi.y;
        wi_perp_u.y = wi.x; 
    }
    assert(abs(dot(wi_perp_u,wi)) <= 1e-5);
    vec3 wi_perp_v = cross(wi, wi_perp_u);

    vec3 ortho_cam_origin = wi * sampling_dist;


    bool continue_sampling = true;
    do {
        
        if (!fill_for_visualisation) {
            // Randomly select one of the cube we will intersect BEFORE we perform the
            // rejection sampling
            // we randomly select inward or outward dilation depending on their ratio
            if (lcg_randomf(lcg) < voxel_extent_dilation_outward / (voxel_extent_dilation_inward + voxel_extent_dilation_outward)) {
                // sample outward
                sampled_voxel_extent_dilation =
                    lcg_randomf(lcg) *
                        (voxel_extent_dilation_outward - voxel_extent_min_dilation) +
                    voxel_extent_min_dilation;  // in [min_dilation_eps; +outward_dilation]
            } else {
                // sample inward
                sampled_voxel_extent_dilation =
                    -(lcg_randomf(lcg) *
                          (voxel_extent_dilation_inward - voxel_extent_min_dilation) +
                      voxel_extent_min_dilation);  // in [-inward_dilation, -min_dilation_eps]
            }
        } else {
            sampled_voxel_extent_dilation = voxel_extent_min_dilation;
        }

        // Perform rejection sampling of the projected area;
        // sampling_dist has to be at least larger than the diagonal of the cube (sqrt(3))
        while (true) {
            // offset by maximum the voxel's diagonal on the projection plane at sqrt(3.0)
            // distance
            vec2 offset = (vec2(lcg_randomf(lcg), lcg_randomf(lcg)) - 0.5f) *
                            sampling_dist;  // > [-0.5 * sqrt(3)/2 ; + 0.5 * sqrt(3.0)/2]
            vec3 visibility_ray_origin =
                ortho_cam_origin + wi_perp_u * offset.x + wi_perp_v * offset.y;

            // Intersection with random aabb, any in [-0.5*(1-eps),0.5*(1 + eps)]
            float t_near, t_far;
            if (intersect_aabb(visibility_ray_origin,
                                ortho_ray_dir,
                                glm::vec3(0.f),
                                1.0f + sampled_voxel_extent_dilation,
                                t_near,
                                t_far)) {
                assert(t_near != -1e20 && t_far != 1e20f);
                local_pi_po[0] = visibility_ray_origin + ortho_ray_dir * t_near;
                local_pi_po[1] = visibility_ray_origin + ortho_ray_dir * t_far;
                break;
            }
        }

        for (size_t p_dim = 0; p_dim < 2; p_dim++) {
            assert(abs(max_component(abs(local_pi_po[p_dim])) -
                       0.5f * (1.f + sampled_voxel_extent_dilation)) < 1e-5);

            pi_po[p_dim] = voxel.center + local_pi_po[p_dim] * voxel.extent;
        }

        if(increase_boundary_density){
            vec3 middle_p = 0.5f * (local_pi_po[0] + local_pi_po[1]);
            vec3 distance_to_center = middle_p / (1+sampled_voxel_extent_dilation);
            assert(length(distance_to_center) <= sqrt(3.f)*0.5f);
            const float unit_manhattan_dist = min_component(abs(distance_to_center)) * 2.0f;
            assert(unit_manhattan_dist >= 0.f && unit_manhattan_dist <= 1.f);
            if (lcg_randomf(lcg) < unit_manhattan_dist * 1.5f + 0.1f) {
                continue_sampling = false;
            }
        }else{
            continue_sampling = false;
        }
    } while (continue_sampling);


    vec3 local_pi;
    vec3 local_po;
    get_deterministic_local_pi_po(local_pi_po,local_pi,local_po);

    vec3 ray_origin = voxel.center + local_pi * voxel.extent; 
    vec3 ray_dir = normalize(local_po - local_pi);

    bool exchange = local_pi_po[0] == local_pi;
    vec3 pi = exchange ? pi_po[0] : pi_po[1];
    vec3 po = exchange ? pi_po[1] : pi_po[0];

    // point on the sphere or cube
    queries[i].origin = ray_origin;
    queries[i].voxel_center = voxel.center;
    queries[i].voxel_extent = voxel.extent * (1.f + max(sampled_voxel_extent_dilation,voxel_bound_bias));
    queries[i].dir = ray_dir;
    queries[i].t_max = 1e20;
 
    if (input_data != nullptr){
        vec3 input_pi = (pi - voxel_grid_aabb_const.min) / voxel_grid_aabb_const.extent();
        vec3 input_po = (po - voxel_grid_aabb_const.min) / voxel_grid_aabb_const.extent();
        const float min_dilation_scaling = 1.f / (1.f + voxel_extent_min_dilation);

        // Setting pi
        input_data[input_idx + 0] = fill_for_visualisation ? local_pi.x * min_dilation_scaling : input_pi.x;
        input_data[input_idx + 1] = fill_for_visualisation ? local_pi.y * min_dilation_scaling : input_pi.y;
        input_data[input_idx + 2] = fill_for_visualisation ? local_pi.z * min_dilation_scaling : input_pi.z;

        // Setting po
        input_data[input_idx + 3] = fill_for_visualisation ? local_po.x * min_dilation_scaling : input_po.x;
        input_data[input_idx + 4] = fill_for_visualisation ? local_po.y * min_dilation_scaling : input_po.y;
        input_data[input_idx + 5] = fill_for_visualisation ? local_po.z * min_dilation_scaling : input_po.z;

    }
}

__global__ void record_ray_query_results_visibility_optim(int batch_size,
                                                      RenderRayQueryResult const *ray_results,
                                                      float *__restrict__ result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    uint32_t output_idx = i * visibility_n_output_dims;

    const RenderRayQueryResult &ray_result = ray_results[i];

    result[output_idx] = ray_result.result.a;
}

__global__ void record_reference_visibility_estimates(int batch_size,
                                                      RenderRayQuery const *ray_queries,
                                                      RenderRayQueryResult const *ray_results,
                                                      float *visualisation_input_batch,
                                                      float *ref_output_accumulated,
                                                      uvec3 sampling_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= batch_size)
        return;

    const RenderRayQueryResult &ray_result = ray_results[i];

    uint32_t input_idx = i * visibility_n_input_dims;

    vec3 local_pi;
    local_pi.x = visualisation_input_batch[input_idx + 0];
    local_pi.y = visualisation_input_batch[input_idx + 1];
    local_pi.z = visualisation_input_batch[input_idx + 2];
    vec3 local_po;
    local_po.x = visualisation_input_batch[input_idx + 3];
    local_po.y = visualisation_input_batch[input_idx + 4];
    local_po.z = visualisation_input_batch[input_idx + 5];

    uint32_t output_idx = compute_inference_idx_for_visibility(
        local_pi, local_po, sampling_res, (visibility_n_output_dims + 1));

    atomicAdd(&ref_output_accumulated[output_idx + 0], ray_result.result.a);
    atomicAdd(&ref_output_accumulated[output_idx + 1], 1.0);
}

__device__ void generate_ray_for_pixel_pos(vec2 pixel_pos,
                                           vec3 cam_origin,
                                           vec3 cam_dir,
                                           vec3 cam_up,
                                           float aspect_ratio,
                                           vec3 &ray_origin,
                                           vec3 &ray_dir)
{
    auto theta = glm::radians(65.f);
    auto h = tan(theta / 2.f);
    auto viewport_height = 2.f * h;
    auto viewport_width = aspect_ratio * viewport_height;

    auto w = normalize(-cam_dir);
    auto u = normalize(cross(cam_up, w));
    auto v = cross(w, u);

    vec3 horizontal = viewport_width * u;
    vec3 vertical = viewport_height * v;
    vec3 lower_left_corner = cam_origin - horizontal / 2.f - vertical / 2.f - w;

    vec2 sample_pos = pixel_pos / vec2((float)fb_width, (float)fb_height);

    ray_origin = cam_origin;
    ray_dir = normalize(lower_left_corner + sample_pos.x * horizontal +
                        (1.0f - sample_pos.y) * vertical - ray_origin);
    return;
}

__global__ void raymarch_grid(uint32_t frame_buffer_elements,
                              vec3 cam_origin,
                              vec3 cam_dir,
                              vec3 cam_up,
                              float aspect,
                              const nanovdb::FloatGrid *d_grid,
                              bool debug_grid_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= frame_buffer_elements || d_grid == nullptr)
        return;

    int ix = i % fb_width;
    int iy = i / fb_width;

    if (ix < 0 || ix >= fb_width)
        return;
    if (iy < 0 || iy >= fb_height)
        return;

    vec2 pixel_pos = vec2((float)ix, (float)iy);

    vec3 ray_origin, ray_dir;
    generate_ray_for_pixel_pos(pixel_pos,cam_origin,cam_dir,cam_up,aspect,ray_origin,ray_dir);

    // Change this to float for floating point precision
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    const RayT world_ray(Vec3T(ray_origin.x, ray_origin.y, ray_origin.z),
                         Vec3T(ray_dir.x, ray_dir.y, ray_dir.z));
    RayT index_ray = world_ray.worldToIndexF(*d_grid);
   
    auto acc = d_grid->tree().getAccessor();
    vec3 color(0.0f,0.0f,0.0f);

    nanovdb::Coord voxel_idx;
    RealT t_index_ray_start = 0.f;
    RealT t_index_ray_end = 0.f;
    float grid_value;
    if (first_active_voxel_intersection(index_ray,
                                        acc,
                                        voxel_idx,
                                        t_index_ray_start,
                                        t_index_ray_end,
                                        grid_value)) {
        float cube_t = (t_index_ray_start + t_index_ray_end) * 0.5f;
        ivec3 cube_voxel_idx = ivec3(index_ray(cube_t)[0],index_ray(cube_t)[1],index_ray(cube_t)[2]);

        // We might end up on the edge of a voxel which in the worst case can mean 2 dimensions shifted
        int shifted_voxels = 0;
        for(size_t dim = 0; dim < 3; dim++)
            if (int(cube_voxel_idx[dim]) != int(voxel_idx[dim]))
                shifted_voxels++;
        if(shifted_voxels > 2){ 
            printf(
                "cube_voxel_idx : %u %u %u , voxel_idx : %u %u "
                "%u\n",
                cube_voxel_idx[0],
                cube_voxel_idx[1],
                cube_voxel_idx[2],
                voxel_idx[0],
                voxel_idx[1],
                voxel_idx[2]);
                float4 data = {1.f,0.f,1.f,1.0};
            surf2Dwrite(data, debug_buffer, ix * 4 * sizeof(float), iy, cudaBoundaryModeZero);
            return;
        }

        if (!debug_grid_data) {
            color.x = voxel_idx[0] / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
            color.y = voxel_idx[1] / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
            color.z = voxel_idx[2] / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
        } else {
            color = jet_colormap(grid_value);
            // color.x = grid_voxel_idx.x / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
            // color.y = grid_voxel_idx.y / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
            // color.z = grid_voxel_idx.z / float(VOXEL_GRID_MAX_RES() >> current_lod_const);
        }
    }

    float4 data = {color.x,color.y,color.z,1.0};
    surf2Dwrite(data, debug_buffer, ix * 4 * sizeof(float), iy, cudaBoundaryModeZero);
}

__global__ void raymarch_grid_for_inference_valid_samples(
    uint32_t frame_buffer_elements,
    vec3 cam_origin,
    vec3 cam_dir,
    vec3 cam_up,
    uint32_t active_voxel_count,
    uint32_t inference_total_sample_count,
    float aspect,
    const nanovdb::FloatGrid *d_grid,
    uint32_t *sparse_voxel_lod_valid_inferences)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= frame_buffer_elements || d_grid == nullptr || sparse_voxel_lod_valid_inferences == nullptr)
        return;

    int ix = i % fb_width;
    int iy = i / fb_width;

    if (ix < 0 || ix >= fb_width)
        return;
    if (iy < 0 || iy >= fb_height)
        return;

    vec2 pixel_pos = vec2((float)ix, (float)iy);

    vec3 ray_origin, ray_dir;
    generate_ray_for_pixel_pos(pixel_pos,cam_origin,cam_dir,cam_up,aspect,ray_origin,ray_dir);

    // Change this to float for floating point precision
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    const RayT world_ray(Vec3T(ray_origin.x, ray_origin.y, ray_origin.z),
                         Vec3T(ray_dir.x, ray_dir.y, ray_dir.z));
    RayT index_ray = world_ray.worldToIndexF(*d_grid);
   
    auto acc = d_grid->tree().getAccessor();

    nanovdb::Coord voxel_idx;
    RealT t_index_ray_start = 0.f;
    RealT t_index_ray_end = 0.f;
    uint32_t accumulated = 0;
    accumulate_samples_along_ray(index_ray,
                                        acc,
                                        voxel_idx,
                                        t_index_ray_start,
                                        t_index_ray_end,
                                        sparse_voxel_lod_valid_inferences,
                                        current_lod_const,
                                        accumulated);

    vec3 color = vec3(0.0,0.0,0.0);
    if(accumulated != 0){
        float value = active_voxel_count * float(accumulated) / inference_total_sample_count;
        color = jet_colormap(value);
    }

    float4 data = {color.x,color.y,color.z,1.0};
    surf2Dwrite(data, debug_buffer, ix * 4 * sizeof(float), iy, cudaBoundaryModeZero);
}

__global__ void fill_debug_framebuffer_with_throughput_data(uint32_t frame_buffer_elements,
                                                            float *throughput_data,
                                                            uvec2 sampling_res,
                                                            bool to_ldr,
                                                            bool is_reference_data,
                                                            bool log_learning)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(i < frame_buffer_elements))
        return;

    int fb_ix = i % fb_width;
    int fb_iy = i / fb_width;

    int im_width = sampling_res.x * sampling_res.y ;
    int im_height = sampling_res.x * sampling_res.y;
    
    int image_ix = (int)((fb_ix / (float)fb_width) * im_width);
    int image_iy = (int)((fb_iy / (float)fb_height) * im_height);

    assert(image_ix < im_width);
    assert(image_iy < im_height);

    auto wo_xy = coord_1d_to_2d(image_ix, sampling_res.x, sampling_res.y);
    auto wi_xy = coord_1d_to_2d(image_iy, sampling_res.x, sampling_res.y);

    int output_dims = output_dims = is_reference_data ? throughput_n_output_dims + 1 : throughput_n_output_dims;
 
    // uint32_t output_idx = (image_iy * im_width + image_ix) * output_dims;
    uint32_t output_idx = coord_4d_to_1d(wo_xy.x,
                                         wo_xy.y,
                                         wi_xy.x,
                                         wi_xy.y,
                                         sampling_res.x,
                                         sampling_res.y,
                                         sampling_res.x,
                                         sampling_res.y) * output_dims;

    float4 output;
    float weight = 1.f;
    if (is_reference_data) {
        // Divide by weight or output magenta color
        if (!(throughput_data[output_idx + output_dims - 1] > 0.0))
            weight = 0.f;
        else
            weight = 1.f / throughput_data[output_idx + output_dims - 1];
    }

    output.x = throughput_data[output_idx + 0] * weight;
    output.y = throughput_data[output_idx + 1] * weight;
    output.z = throughput_data[output_idx + 2] * weight;

    if(log_learning){
        output.x = expf(output.x) - 1.f;
        output.y = expf(output.y) - 1.f;
        output.z = expf(output.z) - 1.f;
    }

    if (to_ldr) {
        output.x = powf(fmaxf(fminf(output.x, 1.0f), 0.0f), 1.0f / 2.2f);
        output.y = powf(fmaxf(fminf(output.y, 1.0f), 0.0f), 1.0f / 2.2f);
        output.z = powf(fmaxf(fminf(output.z, 1.0f), 0.0f), 1.0f / 2.2f);
    }
    output.w = 1.0f;

    surf2Dwrite(output, debug_buffer, fb_ix * 4 * sizeof(float), fb_iy, cudaBoundaryModeZero);
}

__global__ void fill_debug_framebuffer_with_visibility_data(uint32_t frame_buffer_elements,
                                                       float *visibility_data,
                                                       uvec3 sampling_res,
                                                       bool to_ldr,
                                                       bool is_reference_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(i < frame_buffer_elements))
        return;

    int fb_ix = i % fb_width;
    int fb_iy = i / fb_width;

    int im_width = sampling_res.x * sampling_res.y * sampling_res.z;
    int im_height = sampling_res.x * sampling_res.y * sampling_res.z;
    
    int image_ix = (int)((fb_ix / (float)fb_width) * im_width);
    int image_iy = (int)((fb_iy / (float)fb_height) * im_height);

    assert(image_ix < im_width);
    assert(image_iy < im_height);

    auto pi_xyz = coord_1d_to_3d(image_ix, sampling_res.x, sampling_res.y, sampling_res.z);
    auto po_xyz = coord_1d_to_3d(image_iy, sampling_res.x, sampling_res.y, sampling_res.z);

    int output_dims = is_reference_data ? visibility_n_output_dims + 1 : visibility_n_output_dims;
 
    // uint32_t output_idx = (image_iy * im_width + image_ix) * output_dims;
    uint32_t output_idx = coord_6d_to_1d(pi_xyz.x,
                                         pi_xyz.y,
                                         pi_xyz.z,
                                         po_xyz.x,
                                         po_xyz.y,
                                         po_xyz.z,
                                         sampling_res.x,
                                         sampling_res.y,
                                         sampling_res.z,
                                         sampling_res.x,
                                         sampling_res.y,
                                         sampling_res.z) * output_dims;

    float4 output;
    float weight = 1.f;
    if (is_reference_data) {
        // Divide by weight or output magenta color
        if (!(visibility_data[output_idx + output_dims - 1] > 0.0))
            weight = 0.f;
        else
            weight = 1.f / visibility_data[output_idx + output_dims - 1];
    }

    output.x = visibility_data[output_idx] * weight;
    output.y = visibility_data[output_idx] * weight;
    output.z = visibility_data[output_idx] * weight;

    if (to_ldr) {
        output.x = powf(fmaxf(fminf(output.x, 1.0f), 0.0f), 1.0f / 2.2f);
        output.y = powf(fmaxf(fminf(output.y, 1.0f), 0.0f), 1.0f / 2.2f);
        output.z = powf(fmaxf(fminf(output.z, 1.0f), 0.0f), 1.0f / 2.2f);
    }
    output.w = 1.0f;

    surf2Dwrite(output, debug_buffer, fb_ix * 4 * sizeof(float), fb_iy, cudaBoundaryModeZero);
}

NEURAL_LOD_LEARNING_NAMESPACE_END