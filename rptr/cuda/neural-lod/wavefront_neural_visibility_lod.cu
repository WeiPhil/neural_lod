#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>

#define INCLUDE_BASE_WAVEFRONT_NEURAL_PT
#include "../wavefront_neural_pt.cuh"  // Wavefront PT device environment

#include "neural_lod_learning_common.cuh"
#include "neural_lod_learning_grid.cuh"
#include "wavefront_neural_visibility_lod.h"
#include "../../imstate.h"

namespace clsl {
#define inline __device__ inline

// todo: more GLSL includes

#undef inline
}

namespace clsl {

__constant__ glm::vec3 visibility_color_const;
__constant__ glm::vec3 background_color_const;

namespace neural_visibility {

    __global__ void compact_neural_data(
        const uint32_t n_elements,
        NeuralBounceQuery *src_bounce_queries, NeuralBounceResult* src_bounce_results,
        NeuralBounceQuery *dst_bounce_queries, NeuralBounceResult* dst_bounce_results,
        NeuralBounceQuery *dst_final_bounce_queries, NeuralBounceResult* dst_final_bounce_results,
        uint32_t* active_counter, uint32_t* final_counter,
        bool last_inference
    ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        NeuralBounceQuery& src_bounce_query = src_bounce_queries[i];

        if(src_bounce_query.path_state == ACTIVE_PATH && !last_inference){
            uint32_t idx = atomicAdd(active_counter, 1);
            dst_bounce_queries[idx] = src_bounce_query;
            dst_bounce_results[idx] = src_bounce_results[i];
        }else{
            assert(last_inference || src_bounce_query.path_state == ESCAPED_PATH || src_bounce_query.path_state == ACTIVE_PATH_WITH_INTERSECTION);
            // Either an escaped ray, or an intersection was found
            uint32_t idx = atomicAdd(final_counter, 1);
            dst_final_bounce_queries[idx] = src_bounce_query;
            dst_final_bounce_results[idx] = src_bounce_results[i];
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

    __global__ void visibility_tracing(uint32_t n_elements,
                                       NeuralBounceQuery *bounce_queries,
                                       NeuralBounceResult *bounce_results,
                                       bool display_needed_inferences,
                                       uint32_t max_visibility_inferences)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        NeuralBounceQuery &neural_query = bounce_queries[i];
        NeuralBounceResult &neural_result = bounce_results[i];

        // If not then we should have processed it before
        assert(neural_query.path_state != INACTIVE_PATH);

        int path_state = neural_query.path_state;

        if (display_needed_inferences) {
            glm::vec3 needed_inference_color = jet_colormap(neural_result.needed_inferences /
                                                            float(max_visibility_inferences));
            neural_result.illumination = needed_inference_color;
            neural_query.path_state = INACTIVE_PATH;
            return;
        } else if (path_state == ESCAPED_PATH) {
            neural_result.illumination = clsl::background_color_const;
            neural_query.path_state = INACTIVE_PATH;
            return;
        } else if (path_state == ACTIVE_PATH) {
            neural_result.illumination = vec3(1, 0, 1);
            neural_query.path_state = INACTIVE_PATH;
            return;
        }

        assert(path_state == ACTIVE_PATH_WITH_INTERSECTION);

        neural_lod_learning::NeuralLodLearning::Voxel voxel =
            compute_lod_voxel(neural_result.lod_voxel_idx,
                              voxel_grid_aabb_const,
                              voxel_size_const,
                              current_lod_const);

        float depth_estimate =
            length(view_params.cam_pos - voxel.center) / clsl::max_cam_to_aabox_length_const;

        neural_result.illumination =
            clsl::visibility_color_const * (1.f - max(depth_estimate, 0.0f));

        neural_query.path_state = INACTIVE_PATH;
        return;
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

        NeuralBounceQuery &neural_query = bounce_queries[i];
        NeuralBounceResult &neural_result = bounce_results[i];

        // Otherwise we are doing useless work
        assert(neural_query.path_state == ACTIVE_PATH);

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
        
        vec3 local_pi = index_to_local_voxel_pos(
            voxel_near_pos, neural_result.lod_voxel_idx, clamp_to_range);
        vec3 local_po = index_to_local_voxel_pos(
            voxel_far_pos, neural_result.lod_voxel_idx, clamp_to_range);

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

    __global__ void check_visibility(uint32_t n_elements,
                                     NeuralBounceQuery *bounce_queries,
                                     NeuralBounceResult *bounce_results,
                                     uint32_t visibility_inference,
                                     float *__restrict__ visibility_output_data,
                                     bool stochastic_threshold,
                                     uint32_t max_visibility_inferences)
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
        if (stochastic_threshold) {
            occluded =
                visibility_output_data[output_idx] >= RANDOM_FLOAT1(neural_query.rng, -1);
        } else {
            // visibility is 1 if occluded, 0 if non-occluded
            occluded = visibility_output_data[output_idx] >= neural_result.voxel_threshold;
        }

        if (occluded) {
            // We have a real hit, stop evaluating this path
            neural_query.path_state = ACTIVE_PATH_WITH_INTERSECTION;
        }
    }

}  // namespace neural_throughput_visibility

}  // namespace clsl

WavefrontNeuralVisibilityLod::WavefrontNeuralVisibilityLod(RenderCudaBinding *backend)
    : WavefrontNeuralPT(backend)
{
}

WavefrontNeuralVisibilityLod::~WavefrontNeuralVisibilityLod() {}

std::string WavefrontNeuralVisibilityLod::name() const {
    return "Wavefront Neural Visibility Lod Extension";
}

void WavefrontNeuralVisibilityLod::update_scene_from_backend(const Scene& scene) {
    this->WavefrontNeuralPT::update_scene_from_backend(scene);
}

void WavefrontNeuralVisibilityLod::release_mapped_display_resources() {
    this->WavefrontNeuralPT::release_mapped_display_resources();
}

void WavefrontNeuralVisibilityLod::initialize(const int fb_width, const int fb_height)
{
    this->WavefrontNeuralPT::initialize(fb_width, fb_height);

    if(nn == nullptr){
        nn = std::make_unique<NeuralData>();
        nn->active_counter = tcnn::GPUMemory<uint32_t>(1);
        nn->final_counter = tcnn::GPUMemory<uint32_t>(1);
    }
}

void WavefrontNeuralVisibilityLod::setup_constants(
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

    checkCuda(cudaMemcpyToSymbol(clsl::visibility_color_const,
                                 &params.visibility_color[0],
                                 sizeof(params.visibility_color)));
    checkCuda(cudaMemcpyToSymbol(clsl::background_color_const,
                                 &params.background_color[0],
                                 sizeof(params.background_color)));
}

void WavefrontNeuralVisibilityLod::voxel_tracing(tcnn_network *visibility_network,
                                                 const nanovdb::FloatGrid **d_grids)
{
    cudaStream_t default_stream = 0;

    uint32_t n_need_visibility_inference = m_num_rays_initialised;

    checkCuda(cudaMemsetAsync(nn->final_counter.data(), 0, sizeof(uint32_t), default_stream));

#define DEBUGGING
    camera_raygen(m_wavefront_neural_data[0].queries, m_wavefront_neural_data[0].results);

#ifdef DEBUGGING
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
#endif

    clsl::WavefrontNeuralBounceData* neural_data = &m_wavefront_neural_data[0];
    clsl::WavefrontNeuralBounceData* neural_data_prev = &m_wavefront_neural_data[1];

    for (uint32_t inference_idx = 0; inference_idx <= params.max_visibility_inferences; inference_idx++) {

        // First raymarch to check primary visibility
        tcnn::linear_kernel(clsl::neural_visibility::raymarch,
                                    0,
                                    0,
                                    n_need_visibility_inference,
                                    neural_data->queries,
                                    neural_data->results,
                                    0,
                                    d_grids);
#ifdef DEBUGGING
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
#endif

        // Prepare for compaction
        neural_data = &m_wavefront_neural_data[(inference_idx + 1) % 2];
		neural_data_prev = &m_wavefront_neural_data[inference_idx % 2];

		// Compact rays
		{
			checkCuda(cudaMemsetAsync(nn->active_counter.data(), 0, sizeof(uint32_t), default_stream));
			tcnn::linear_kernel(clsl::neural_visibility::compact_neural_data, 0, default_stream,
				n_need_visibility_inference,
				neural_data_prev->queries,neural_data_prev->results,
                neural_data->queries, neural_data->results,
                m_wavefront_neural_data_intersected.queries,m_wavefront_neural_data_intersected.results,
				nn->active_counter.data(), nn->final_counter.data(),
                inference_idx == params.max_visibility_inferences - 1
			);
			checkCuda(cudaMemcpyAsync(&n_need_visibility_inference, nn->active_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
			checkCuda(cudaStreamSynchronize(default_stream));
		}

        if (n_need_visibility_inference == 0) {
			break;
		}

        // Prepare right sized inference buffers
        uint32_t n_padded_width =
            tcnn::next_multiple(n_need_visibility_inference, tcnn::batch_size_granularity);

        tcnn::GPUMatrix<float> visibility_inputs(m_visibility_network_inputs,
                                                 neural_lod_learning::visibility_n_input_dims,
                                                 n_padded_width);
        tcnn::GPUMatrix<float> visibility_outputs(
            m_visibility_network_outputs,
            neural_lod_learning::visibility_n_output_dims,
            n_padded_width);

        tcnn::linear_kernel(clsl::neural_visibility::generate_visibility_inference_data,
                            0,
                            default_stream,
                            n_need_visibility_inference,
                            neural_data->queries,
                            neural_data->results,
                            visibility_inputs.data(),
                            params.inference_min_extent_dilation);

#ifdef DEBUGGING
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
#endif


        // Compute Inference
        visibility_network->inference(default_stream, visibility_inputs, visibility_outputs);

        tcnn::linear_kernel(clsl::neural_visibility::check_visibility,
                            0,
                            default_stream,
                            n_need_visibility_inference,
                            neural_data->queries,
                            neural_data->results,
                            inference_idx,
                            visibility_outputs.data(),
                            params.stochastic_threshold,
                            params.max_visibility_inferences);

#ifdef DEBUGGING
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
#endif        

    }

    uint32_t n_processed;
	checkCuda(cudaMemcpyAsync(&n_processed, nn->final_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
	checkCuda(cudaStreamSynchronize(default_stream));
    
    // Sanity check
    assert(n_processed == m_num_rays_initialised);
    
    tcnn::linear_kernel(clsl::neural_visibility::visibility_tracing,
                        0,
                        default_stream,
                        n_processed,
                        m_wavefront_neural_data_intersected.queries,
                        m_wavefront_neural_data_intersected.results,
                        params.display_needed_inferences,
                        params.max_visibility_inferences);

#ifdef DEBUGGING
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
#endif

    accumulate_results(m_wavefront_neural_data_intersected.queries,
                       m_wavefront_neural_data_intersected.results);
}

void WavefrontNeuralVisibilityLod::render(bool reset_accumulation)
{
    render(reset_accumulation,
           nullptr,
           nullptr,
           0,
           0.f);
}

void WavefrontNeuralVisibilityLod::render(
    bool reset_accumulation,
    neural_lod_learning::NeuralLodLearning::VisibilityNN *visibility_neural_net,
    neural_lod_learning::NeuralLodLearning::VoxelGrid* voxel_grid,
    int current_lod,
    float voxel_extent_dilation_outward)
{
    if (reset_accumulation) {
        sample_offset += accumulated_spp;
        accumulated_spp = 0;
    }

    // Sync cpu side current_lod
    params.current_lod = current_lod;

    if (!visibility_neural_net || !voxel_grid)
        return;

    // Pre-compute approximate max distance for depth map
    float max_cam_to_aabox_length = -1.f;
    auto extent = voxel_grid->aabb.extent();
    for (size_t a = 0; a < 2; a++)
        for (size_t b = 0; b < 2; b++)
            for (size_t c = 0; c < 2; c++) {
                float lenght =
                    glm::length(backend->backend->camera.pos -
                                glm::vec3(voxel_grid->aabb.min[0] + a * extent[0],
                                            voxel_grid->aabb.min[1] + b * extent[1],
                                            voxel_grid->aabb.min[2] + c * extent[2]));
                max_cam_to_aabox_length = max(lenght, max_cam_to_aabox_length);
            }
    setup_constants(voxel_grid->aabb, voxel_grid->voxel_size, current_lod,voxel_extent_dilation_outward, max_cam_to_aabox_length);

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

    for (int spp = 0; spp < backend->backend->params.batch_spp; ++spp) {
        voxel_tracing(visibility_neural_net->network.get(), d_grids);
        ++accumulated_spp;
    }

    checkCuda(cudaFree(d_grids));
}