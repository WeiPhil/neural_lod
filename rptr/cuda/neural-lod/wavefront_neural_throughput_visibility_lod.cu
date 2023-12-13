#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include "util/tinyexr.h"

#define INCLUDE_BASE_WAVEFRONT_NEURAL_PT
#include "../wavefront_neural_pt.cuh"  // Wavefront PT device environment

#include "../../imstate.h"
#include "neural_lod_learning_common.cuh"
#include "neural_lod_learning_grid.cuh"
#include "wavefront_neural_throughput_visibility_lod.h"

#include <glm/gtx/rotate_vector.hpp>

#include <chrono>
using namespace std::chrono;

namespace clsl {
#define inline __device__ inline

// todo: more GLSL includes

#undef inline
}

namespace clsl {

__constant__ glm::vec3 envmap_color_const;
__constant__ glm::vec3 background_color_const;
__constant__ cudaSurfaceObject_t envmap_texture_const;

namespace neural_throughput_visibility {

    inline __device__ void process_final_bounce(NeuralBounceQuery &neural_query,
                                                NeuralBounceResult &neural_result,
                                                int max_depth,
                                                int max_visibility_inferences,
                                                int bounce,
                                                bool use_envmap,
                                                float envmap_rotation)
    {
#ifdef STATS_NUM_BOUNCES
        if (view_params.rparams.output_moment == 1) {
            neural_result.illumination = vec3(bounce);
            neural_query.path_state = INACTIVE_PATH;
            return;
        }
#endif

        if (neural_query.path_state == ESCAPED_PATH) {
            if (use_envmap) {
                glm::vec3 dir = neural_query.dir;
                glm::mat4 rotation = glm::rotate(glm::mat4(1.f),
                                                 envmap_rotation * float(M_PI) / 180.f,
                                                 glm::vec3(0.0, 1.0, 0.0));
                dir = glm::vec3(rotation * glm::vec4(dir, 1.0));
                // [-0.5,0.5] -> [0,1]
                float envmap_u = atan2(dir.x, -dir.z) * M_1_PI * 0.5f + 0.5f;
                float envmap_v = safe_acos(dir.y) * M_1_PI;
                float4 val = tex2D<float4>(clsl::envmap_texture_const, envmap_u, envmap_v);
                // HACK : clamping to 100.f to avoid fireflies due to strong sun
                glm::vec3 envmap_val =
                    glm::min(glm::vec3(100.f), glm::vec3(val.x, val.y, val.z));
                neural_result.illumination =
                    bounce == 0 ? envmap_val : neural_result.path_throughput * envmap_val;
            } else {
                neural_result.illumination =
                    bounce == 0 ? clsl::background_color_const
                                : neural_result.path_throughput * clsl::envmap_color_const;
            }
        } else {
            assert(neural_query.path_state == ACTIVE_PATH ||
                   neural_query.path_state == ACTIVE_PATH_WITH_INTERSECTION ||
                   neural_query.path_state == INACTIVE_PATH);
        }

        neural_query.path_state = INACTIVE_PATH;
        return;
    }

    __global__ void compact_for_visibility_and_accumulate(
        uint32_t n_elements,
        int sample_idx,
        NeuralBounceQuery *src_bounce_queries,
        NeuralBounceResult *src_bounce_results,
        NeuralBounceQuery *dst_bounce_queries,
        NeuralBounceResult *dst_bounce_results,
        NeuralBounceQuery *dst_intersection_bounce_queries,
        NeuralBounceResult *dst_intersection_bounce_results,
        uint32_t *active_counter,
        uint32_t *active_intersection_counter,
        bool last_inference,
        bool last_bounce,
        int max_depth,
        int max_visibility_inferences,
        int bounce,
        bool use_envmap,
        float envmap_rotation)
    {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements)
            return;

        NeuralBounceQuery &src_bounce_query = src_bounce_queries[i];
        NeuralBounceResult &src_bounce_result = src_bounce_results[i];

        if (src_bounce_query.path_state == ACTIVE_PATH && !last_inference) {
            uint32_t idx = atomicAdd(active_counter, 1);
            dst_bounce_queries[idx] = src_bounce_query;
            dst_bounce_results[idx] = src_bounce_result;
        } else if (src_bounce_query.path_state == ACTIVE_PATH_WITH_INTERSECTION &&
                   !last_bounce) {
            uint32_t idx = atomicAdd(active_intersection_counter, 1);
            // We want to evaluate those intersections before the next bounce
            dst_intersection_bounce_queries[idx] = src_bounce_query;
            dst_intersection_bounce_results[idx] = src_bounce_result;
        } else {
            // We don't need to evaluate the throughput network, directly splate to the
            // framebuffer
            assert(last_inference || last_bounce ||
                   src_bounce_query.path_state == ESCAPED_PATH ||
                   src_bounce_query.path_state == INACTIVE_PATH);

            // Either an escaped ray or we reachde the maximal number of inferences or bounces
            clsl::neural_throughput_visibility::process_final_bounce(src_bounce_query,
                                                                     src_bounce_result,
                                                                     max_depth,
                                                                     max_visibility_inferences,
                                                                     bounce,
                                                                     use_envmap,
                                                                     envmap_rotation);
            accumulate_neural_bounce(sample_idx, src_bounce_query, src_bounce_result);
        }
    }

    __global__ void compact_for_next_bounce_and_accumulate(
        uint32_t n_elements,
        int sample_idx,
        NeuralBounceQuery *src_intersection_bounce_queries,
        NeuralBounceResult *src_intersection_bounce_results,
        NeuralBounceQuery *dst_next_bounce_queries,
        NeuralBounceResult *dst_next_bounce_results,
        uint32_t *active_counter,
        int max_depth,
        int max_visibility_inferences,
        int bounce,
        bool use_envmap,
        float envmap_rotation)
    {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements)
            return;

        NeuralBounceQuery &src_intersection_bounce_query = src_intersection_bounce_queries[i];
        NeuralBounceResult &src_intersection_bounce_result =
            src_intersection_bounce_results[i];

        if (src_intersection_bounce_query.path_state == ACTIVE_PATH) {
            // We want to continue that path
            uint32_t idx = atomicAdd(active_counter, 1);
            dst_next_bounce_queries[idx] = src_intersection_bounce_query;
            dst_next_bounce_results[idx] = src_intersection_bounce_results[i];
        } else {
            // The path was killed
            assert(src_intersection_bounce_query.path_state == INACTIVE_PATH);

            // Either an escaped ray or we reachde the maximal number of inferences or bounces
            clsl::neural_throughput_visibility::process_final_bounce(
                src_intersection_bounce_query,
                src_intersection_bounce_result,
                max_depth,
                max_visibility_inferences,
                bounce,
                use_envmap,
                envmap_rotation);
            accumulate_neural_bounce(
                sample_idx, src_intersection_bounce_query, src_intersection_bounce_result);
        }
    }

    __global__ void raymarch(uint32_t n_elements,
                             NeuralBounceQuery *bounce_queries,
                             NeuralBounceResult *bounce_results,
                             uint32_t visibility_inference,
                             const nanovdb::FloatGrid **d_grids)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        clsl::raymarch_impl(i,
                            bounce_queries,
                            bounce_results,
                            0,
                            visibility_inference,
                            d_grids[current_lod_const]);
    }

    __global__ void process_ray_bounce(uint32_t n_elements,
                                       uint32_t bounce,
                                       NeuralBounceQuery *bounce_queries,
                                       NeuralBounceResult *bounce_results,
                                       float *__restrict__ throughput_output_data,
                                       bool apply_rr,
                                       uint32_t rr_start_bounce,
                                       bool throughput_log_learning)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        NeuralBounceQuery &neural_query = bounce_queries[i];
        NeuralBounceResult &neural_result = bounce_results[i];

        assert(neural_query.path_state == ACTIVE_PATH_WITH_INTERSECTION);

        vec3 throughput_inference =
            clsl::get_throughput_inference(i, throughput_output_data, throughput_log_learning);
        float throughput_inference_pdf_inv = neural_result.throughput_inference_pdf_inv;

        neural_result.path_throughput *= throughput_inference * throughput_inference_pdf_inv;

        bool continue_path = neural_result.path_throughput != vec3(0.0f);

        neural_lod_learning::NeuralLodLearning::Voxel voxel =
            compute_lod_voxel(neural_result.lod_voxel_idx,
                              clsl::voxel_grid_aabb_const,
                              clsl::voxel_size_const,
                              clsl::current_lod_const);

        assert(neural_result.t_voxel_far > neural_result.t_voxel_near);
        const float t_eps = neural_result.t_voxel_near +
                            1e-4 * (neural_result.t_voxel_far - neural_result.t_voxel_near);

        neural_query.origin = neural_query.origin + neural_query.dir * t_eps;
        neural_query.dir = neural_result.throughput_wi;
        neural_result.t_voxel_near = -1.f;
        neural_result.t_voxel_far = -1.f;

        if (apply_rr)
            continue_path =
                continue_path &&
                rr_terminate(
                    bounce, rr_start_bounce, neural_result.path_throughput, neural_query.rng);

        if (continue_path) {
            neural_query.path_state = ACTIVE_PATH;
        } else {
            neural_query.path_state = INACTIVE_PATH;
        }
    }

    __global__ void generate_visibility_inference_data(
        uint32_t n_elements,
        NeuralBounceQuery *bounce_queries,
        NeuralBounceResult *bounce_results,
        float *__restrict__ visibility_input_data,
        float inference_min_extent_dilation)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        clsl::generate_visibility_inference_data_impl(i,
                                                      bounce_queries,
                                                      bounce_results,
                                                      visibility_input_data,
                                                      inference_min_extent_dilation);
    }

    __global__ void generate_throughput_inference_data(
        uint32_t n_elements,
        NeuralBounceQuery *bounce_queries,
        NeuralBounceResult *bounce_results,
        float *__restrict__ throughput_input_data,
        bool use_hg,
        float hg_g)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        clsl::generate_throughput_inference_data_impl(
            i, bounce_queries, bounce_results, throughput_input_data, use_hg, hg_g);
    }

    __global__ void check_visibility(uint32_t n_elements,
                                     NeuralBounceQuery *bounce_queries,
                                     NeuralBounceResult *bounce_results,
                                     uint32_t visibility_inference,
                                     float *__restrict__ visibility_output_data,
                                     bool stochastic_threshold,
                                     uint32_t max_visibility_inferences,
                                     bool apply_visibility_rr)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        NeuralBounceQuery &neural_query = bounce_queries[i];
        NeuralBounceResult &neural_result = bounce_results[i];

        int path_state = neural_query.path_state;

        // Stop if we are inactive or we reached the target lod intersection
        if (path_state != ACTIVE_PATH) {
            return;
        }

        // Update needed inferences
        neural_result.needed_inferences += 1;

        uint32_t output_idx = i * neural_lod_learning::visibility_n_output_dims;

        bool occluded;
        float visibility_inference_value = visibility_output_data[output_idx];
        if (stochastic_threshold) {
            // rescaled using visibility threshold
            occluded = visibility_inference_value >= RANDOM_FLOAT1(neural_query.rng, -1);
        } else {
            occluded = visibility_inference_value >= neural_result.voxel_threshold;
        }

        if (occluded) {
            // We have a real hit, stop evaluating this path
            neural_query.path_state = ACTIVE_PATH_WITH_INTERSECTION;
        } else if (apply_visibility_rr) {
            if (!clsl::visibility_rr_terminate(visibility_inference_value,
                                               visibility_inference,
                                               neural_result.path_throughput,
                                               neural_query.rng)) {
                neural_query.path_state = INACTIVE_PATH;
            }
        }
    }

}  // namespace neural_throughput_visibility

}  // namespace clsl

WavefrontNeuralThroughputVisibilityLod::WavefrontNeuralThroughputVisibilityLod(
    RenderCudaBinding *backend)
    : WavefrontNeuralPT(backend)
{
}

WavefrontNeuralThroughputVisibilityLod::~WavefrontNeuralThroughputVisibilityLod() {}

std::string WavefrontNeuralThroughputVisibilityLod::name() const
{
    return "Wavefront Neural Throughput and Visibility Lod Extension";
}

void WavefrontNeuralThroughputVisibilityLod::update_scene_from_backend(const Scene &scene)
{
    this->WavefrontNeuralPT::update_scene_from_backend(scene);
}

void WavefrontNeuralThroughputVisibilityLod::release_mapped_display_resources()
{
    this->WavefrontNeuralPT::release_mapped_display_resources();
}

namespace {
tcnn::GPUMemory<float> load_image(const char *filename, int &width, int &height)
{
    float *data;  // width * height * RGBA
    const char *err = nullptr;

    int ret = LoadEXR(&data, &width, &height, filename, &err);
    if (ret != TINYEXR_SUCCESS) {
        width = 4;
        height = 4;
        println(CLL::WARNING, "Envmap %s not found, setting black image",filename);
        tcnn::GPUMemory<float> result(width * height * 4);
        return result;
    }

    tcnn::GPUMemory<float> result(width * height * 4);
    result.copy_from_host(data);
    free(data);  // release memory of image data

    return result;
}
}

void WavefrontNeuralThroughputVisibilityLod::load_envmap()
{
    // First step: load an image that we'd like to learn
    int width, height;
    envmap_data->envmap_image = load_image(params.envmap_filename, width, height);

    println(CLL::INFORMATION, "Reading envmap with size : %i %i", width, height);

    // Create a cuda texture out of this image
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = envmap_data->envmap_image.data();
    resDesc.res.pitch2D.desc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    checkCuda(cudaCreateTextureObject(&envmap_texture, &resDesc, &texDesc, nullptr));
}

void WavefrontNeuralThroughputVisibilityLod::initialize(const int fb_width,
                                                        const int fb_height)
{
    this->WavefrontNeuralPT::initialize(fb_width, fb_height);

    if (nn == nullptr) {
        nn = std::make_unique<NeuralData>();
        nn->active_counter = tcnn::GPUMemory<uint32_t>(1);
        nn->active_intersection_counter = tcnn::GPUMemory<uint32_t>(1);
    }
}

void WavefrontNeuralThroughputVisibilityLod::setup_constants(
    NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
    float voxel_size,
    int current_lod,
    float voxel_extent_dilation_outward,
    float max_cam_to_aabox_length)
{
    this->WavefrontNeuralPT::setup_constants(voxel_grid_aabb,
                                             voxel_size,
                                             current_lod,
                                             voxel_extent_dilation_outward,
                                             max_cam_to_aabox_length);
    this->WavefrontNeuralPT::this_module_setup_constants(voxel_grid_aabb,
                                                         voxel_size,
                                                         current_lod,
                                                         voxel_extent_dilation_outward,
                                                         max_cam_to_aabox_length);

    checkCuda(cudaMemcpyToSymbol(
        clsl::envmap_color_const, &params.envmap_color[0], sizeof(params.envmap_color)));
    checkCuda(cudaMemcpyToSymbol(clsl::background_color_const,
                                 &params.background_color[0],
                                 sizeof(params.background_color)));

    checkCuda(cudaMemcpyToSymbol(
        clsl::envmap_texture_const, &envmap_texture, sizeof(clsl::envmap_texture_const)));
    if (envmap_data == nullptr) {
        envmap_data = std::make_unique<EnvmapData>();
        load_envmap();
    }
}

void WavefrontNeuralThroughputVisibilityLod::voxel_tracing(int sample_idx,
                                                           tcnn_network *visibility_network,
                                                           tcnn_network *throughput_network,
                                                           bool throughput_log_learning,
                                                           const nanovdb::FloatGrid **d_grids)
{
    cudaStream_t default_stream = 0;

    uint32_t n_need_visibility_inference = m_num_rays_initialised;

    uint32_t double_buffer_index = 0;

#define DEBUGGING
    camera_raygen(m_wavefront_neural_data[double_buffer_index].queries,
                  m_wavefront_neural_data[double_buffer_index].results);

#ifdef DEBUGGING
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
#endif

    auto total_duration = 0;
    auto total_frame = 0;

    // Manhattan distance from one corner to another
    uint32_t max_visibility_inferences = (VOXEL_GRID_MAX_RES() >> params.current_lod) * 3;
    for (uint32_t bounce = 0; bounce < params.max_depth; bounce++) {
        // Reset the active intersection_counter
        checkCuda(cudaMemsetAsync(
            nn->active_intersection_counter.data(), 0, sizeof(uint32_t), default_stream));

        total_duration = 0;

        for (uint32_t inference_idx = 0; inference_idx < max_visibility_inferences;
             inference_idx++) {
            auto start = high_resolution_clock::now();

            tcnn::linear_kernel(clsl::neural_throughput_visibility::raymarch,
                                0,
                                0,
                                n_need_visibility_inference,
                                m_wavefront_neural_data[double_buffer_index % 2].queries,
                                m_wavefront_neural_data[double_buffer_index % 2].results,
                                0,
                                d_grids);

#ifdef DEBUGGING
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
#endif
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);

            total_duration += duration.count();

            clsl::WavefrontNeuralBounceData neural_data =
                m_wavefront_neural_data[(double_buffer_index + 1) % 2];
            clsl::WavefrontNeuralBounceData neural_data_prev =
                m_wavefront_neural_data[double_buffer_index % 2];
            double_buffer_index++;

            // Compact rays and splat terminated rays
            {
                checkCuda(cudaMemsetAsync(
                    nn->active_counter.data(), 0, sizeof(uint32_t), default_stream));
                tcnn::linear_kernel(
                    clsl::neural_throughput_visibility::compact_for_visibility_and_accumulate,
                    0,
                    default_stream,
                    n_need_visibility_inference,
                    sample_idx,
                    neural_data_prev.queries,
                    neural_data_prev.results,
                    neural_data.queries,
                    neural_data.results,
                    m_wavefront_neural_data_intersected.queries,
                    m_wavefront_neural_data_intersected.results,
                    nn->active_counter.data(),
                    nn->active_intersection_counter.data(),
                    inference_idx == max_visibility_inferences - 1,
                    bounce == params.max_depth - 1,
                    params.max_depth,
                    max_visibility_inferences,
                    bounce,
                    params.use_envmap,
                    params.envmap_rotation);
                checkCuda(cudaMemcpyAsync(&n_need_visibility_inference,
                                          nn->active_counter.data(),
                                          sizeof(uint32_t),
                                          cudaMemcpyDeviceToHost,
                                          default_stream));
                checkCuda(cudaStreamSynchronize(default_stream));
            }

            if (n_need_visibility_inference == 0) {
                break;
            }

            // We should have exited the loop before!
            assert(inference_idx != max_visibility_inferences - 1);

            // Prepare right sized inference buffers
            uint32_t n_padded_width =
                tcnn::next_multiple(n_need_visibility_inference, tcnn::batch_size_granularity);

            tcnn::GPUMatrix<float> visibility_inputs(
                m_visibility_network_inputs,
                neural_lod_learning::visibility_n_input_dims,
                n_padded_width);
            tcnn::GPUMatrix<float> visibility_outputs(
                m_visibility_network_outputs,
                neural_lod_learning::visibility_n_output_dims,
                n_padded_width);

            tcnn::linear_kernel(
                clsl::neural_throughput_visibility::generate_visibility_inference_data,
                0,
                default_stream,
                n_need_visibility_inference,
                neural_data.queries,
                neural_data.results,
                visibility_inputs.data(),
                params.inference_min_extent_dilation);

#ifdef DEBUGGING
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
#endif

            // Compute Visibility Inference
            visibility_network->inference(
                default_stream, visibility_inputs, visibility_outputs);

            tcnn::linear_kernel(clsl::neural_throughput_visibility::check_visibility,
                                0,
                                default_stream,
                                n_need_visibility_inference,
                                neural_data.queries,
                                neural_data.results,
                                inference_idx,
                                visibility_outputs.data(),
                                params.stochastic_threshold,
                                max_visibility_inferences,
                                params.apply_visibility_rr);

#ifdef DEBUGGING
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
#endif
        }

        assert(n_need_visibility_inference == 0);

        uint32_t n_intersection;
        checkCuda(cudaMemcpyAsync(&n_intersection,
                                  nn->active_intersection_counter.data(),
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost,
                                  default_stream));
        checkCuda(cudaStreamSynchronize(default_stream));

        if (n_intersection == 0) {
            break;
        }

        // We should have exited the loop before!
        assert(bounce != params.max_depth - 1);

        // Prepare right sized inference buffers
        uint32_t n_padded_width =
            tcnn::next_multiple(n_intersection, tcnn::batch_size_granularity);

        tcnn::GPUMatrix<float> throughput_inputs(m_throughput_network_inputs,
                                                 neural_lod_learning::throughput_n_input_dims,
                                                 n_padded_width);
        tcnn::GPUMatrix<float> throughput_outputs(
            m_throughput_network_outputs,
            neural_lod_learning::throughput_n_output_dims,
            n_padded_width);

        tcnn::linear_kernel(
            clsl::neural_throughput_visibility::generate_throughput_inference_data,
            0,
            default_stream,
            n_intersection,
            m_wavefront_neural_data_intersected.queries,
            m_wavefront_neural_data_intersected.results,
            throughput_inputs.data(),
            params.use_hg,
            params.hg_g);

#ifdef DEBUGGING
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
#endif

        // Compute Throughput Inference
        throughput_network->inference(default_stream, throughput_inputs, throughput_outputs);

        // Procees the bounce and proceed to a new bounce
        tcnn::linear_kernel(clsl::neural_throughput_visibility::process_ray_bounce,
                            0,
                            default_stream,
                            n_intersection,
                            bounce,
                            m_wavefront_neural_data_intersected.queries,
                            m_wavefront_neural_data_intersected.results,
                            throughput_outputs.data(),
                            params.apply_rr,
                            params.rr_start_bounce,
                            throughput_log_learning);

#ifdef DEBUGGING
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
#endif

        clsl::WavefrontNeuralBounceData neural_data_prev =
            m_wavefront_neural_data[double_buffer_index % 2];

        // Compact rays for next bounce
        {
            checkCuda(cudaMemsetAsync(
                nn->active_counter.data(), 0, sizeof(uint32_t), default_stream));
            tcnn::linear_kernel(
                clsl::neural_throughput_visibility::compact_for_next_bounce_and_accumulate,
                0,
                default_stream,
                n_intersection,
                sample_idx,
                m_wavefront_neural_data_intersected.queries,
                m_wavefront_neural_data_intersected.results,
                neural_data_prev.queries,
                neural_data_prev.results,
                nn->active_counter.data(),
                params.max_depth,
                max_visibility_inferences,
                bounce,
                params.use_envmap,
                params.envmap_rotation);
            checkCuda(cudaMemcpyAsync(&n_need_visibility_inference,
                                      nn->active_counter.data(),
                                      sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost,
                                      default_stream));
            checkCuda(cudaStreamSynchronize(default_stream));
        }

        // std::cout << "bounce " << bounce << " total : " << total_duration << "micro s" <<
        // std::endl;

        total_frame += total_duration;
    }

    // std::cout << "total_frame " << total_frame << " micro s" << std::endl;
}

void WavefrontNeuralThroughputVisibilityLod::render(bool reset_accumulation)
{
    render(reset_accumulation, nullptr, nullptr, false, nullptr, 0, 0.f);
}

void WavefrontNeuralThroughputVisibilityLod::render(
    bool reset_accumulation,
    neural_lod_learning::NeuralLodLearning::VisibilityNN *visibility_neural_net,
    neural_lod_learning::NeuralLodLearning::ThroughputNN *throughput_neural_net,
    bool throughput_log_learning,
    neural_lod_learning::NeuralLodLearning::VoxelGrid *voxel_grid,
    int current_lod,
    float voxel_extent_dilation_outward)
{
    if (reset_accumulation) {
        sample_offset += accumulated_spp;
        accumulated_spp = 0;
    }

    // Sync cpu side current_lod
    // params.current_lod = current_lod;

    if (!visibility_neural_net || !throughput_neural_net || !voxel_grid)
        return;

    // Pre-compute approximate max distance for depth map
    float max_cam_to_aabox_length;
    auto extent = voxel_grid->aabb.extent();
    for (size_t a = 0; a < 2; a++)
        for (size_t b = 0; b < 2; b++)
            for (size_t c = 0; c < 2; c++) {
                float lenght = glm::length(backend->backend->camera.pos -
                                           glm::vec3(voxel_grid->aabb.min[0] + a * extent[0],
                                                     voxel_grid->aabb.min[1] + b * extent[1],
                                                     voxel_grid->aabb.min[2] + c * extent[2]));
                max_cam_to_aabox_length = max(lenght, max_cam_to_aabox_length);
            }

    params.current_lod = current_lod;

    setup_constants(voxel_grid->aabb,
                    voxel_grid->voxel_size,
                    params.current_lod,
                    voxel_extent_dilation_outward,
                    max_cam_to_aabox_length);

    // Prepare easy access to all grid levels on the gpu by grouping them in a GPU array
    const nanovdb::FloatGrid **d_grids;
    const nanovdb::FloatGrid *h_device_grid_pointers[NUM_LODS()];

    // allocate device pointers to FloatGrid
    checkCuda(cudaMalloc(&d_grids, NUM_LODS() * sizeof(const nanovdb::FloatGrid *)));
    for (size_t i = 0; i < NUM_LODS(); i++) {
        h_device_grid_pointers[i] = voxel_grid->grid_handles[i].deviceGrid<float>();
    }
    checkCuda(cudaMemcpy(d_grids,
                         &h_device_grid_pointers[0],
                         NUM_LODS() * sizeof(nanovdb::FloatGrid *),
                         cudaMemcpyHostToDevice));

    assert(params.current_lod >= 0 && params.current_lod < NUM_LODS());

    for (int spp = 0; spp < backend->backend->params.batch_spp; ++spp) {
        voxel_tracing(accumulated_spp,
                      visibility_neural_net->network.get(),
                      throughput_neural_net->network.get(),
                      throughput_log_learning,
                      d_grids);
        ++accumulated_spp;
    }

    checkCuda(cudaFree(d_grids));
}