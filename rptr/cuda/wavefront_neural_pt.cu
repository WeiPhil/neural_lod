#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#define INCLUDE_BASE_WAVEFRONT_NEURAL_PT
#include "wavefront_neural_pt.cuh"  // Wavefront PT device environment

namespace clsl {

__global__ void camera_raygen(uint32_t n_elements,
                              int sample_idx,
                              int rnd_sample_offset,
                              NeuralBounceQuery *bounce_queries,
                              NeuralBounceResult *bounce_results)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    NeuralBounceQuery &neural_query = bounce_queries[i];
    NeuralBounceResult &neural_result = bounce_results[i];

    int pixel_x = i % fb_width;
    int pixel_y = i / fb_width;

    auto lcg = get_lcg_rng(sample_idx, rnd_sample_offset, i);
    float xf = (float(pixel_x) + lcg_randomf(lcg)) / float(fb_width);
    float yf = (float(pixel_y) + lcg_randomf(lcg)) / float(fb_height);

    neural_query.origin = vec3(view_params.cam_pos);
    neural_query.dir =
        normalize(xf * vec3(view_params.cam_du) + yf * vec3(view_params.cam_dv) +
                  vec3(view_params.cam_dir_top_left));

    float res = float(VOXEL_GRID_MAX_RES() >> current_lod_const);
    vec3 index_origin = res * ((neural_query.origin - voxel_grid_aabb_const.min) /
                               voxel_grid_aabb_const.extent());
    neural_query.origin = index_origin;

    neural_query.path_state = ACTIVE_PATH;
    neural_query.rng = {lcg.state};

    neural_result.lod_voxel_idx = ivec3(-1);
    neural_result.t_voxel_near = -1.f;
    neural_result.t_voxel_far = -1.f;
    neural_result.needed_inferences = 0;
    neural_result.pixel_coord = coord_2d_to_1d(pixel_x, pixel_y, fb_width, fb_height);
    neural_result.illumination = vec3(0.0f, 0.0f, 0.0f);
    neural_result.path_throughput = vec3(1.0f, 1.0f, 1.0f);
}

__global__ void accumulate_neural_results(uint32_t n_elements,
                                          int sample_index,
                                          NeuralBounceQuery *bounce_queries,
                                          NeuralBounceResult *bounce_results)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    NeuralBounceQuery &neural_query = bounce_queries[i];
    NeuralBounceResult &neural_result = bounce_results[i];

    accumulate_neural_bounce(sample_index, neural_query, neural_result);
}

}  // namespace

void WavefrontNeuralPT::initialize(const int fb_width, const int fb_height)
{
    this->WavefrontNeuralPT::release_mapped_display_resources();

    if (m_tcnn_wrapper == nullptr)
        m_tcnn_wrapper = std::make_unique<TcnnWrapper>();

    m_num_rays_initialised = fb_width * fb_height;

    uint32_t n_padded_elements =
        tcnn::next_multiple(m_num_rays_initialised, tcnn::batch_size_granularity);
    uint32_t n_elements = m_num_rays_initialised;
    auto scratch = tcnn::allocate_workspace_and_distribute<
        clsl::NeuralBounceQuery,
        clsl::NeuralBounceResult,  // m_wavefront_neural_data[0]
        clsl::NeuralBounceQuery,
        clsl::NeuralBounceResult,  // m_wavefront_neural_data[1]
        clsl::NeuralBounceQuery,
        clsl::NeuralBounceResult,  // m_wavefront_neural_data_intersected
        float,
        float,                     // visibility inputs and outputs
        float,
        float                      // throughput inputs and outputs
        >(0,
          &m_tcnn_wrapper->scratch_alloc,
          n_elements,
          n_elements,
          n_elements,
          n_elements,
          n_elements,
          n_elements,
          n_padded_elements * neural_lod_learning::visibility_n_input_dims,
          n_padded_elements * neural_lod_learning::visibility_n_output_dims,
          n_padded_elements * neural_lod_learning::throughput_n_input_dims,
          n_padded_elements * neural_lod_learning::throughput_n_output_dims);

    m_wavefront_neural_data[0].set(std::get<0>(scratch), std::get<1>(scratch), n_elements);
    m_wavefront_neural_data[1].set(std::get<2>(scratch), std::get<3>(scratch), n_elements);
    m_wavefront_neural_data_intersected.set(
        std::get<4>(scratch), std::get<5>(scratch), n_elements);

    m_visibility_network_inputs = std::get<6>(scratch);
    m_visibility_network_outputs = std::get<7>(scratch);
    m_throughput_network_inputs = std::get<8>(scratch);
    m_throughput_network_outputs = std::get<9>(scratch);
}

void WavefrontNeuralPT::setup_constants(NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
                                        float voxel_size,
                                        int current_lod,
                                        float voxel_extent_dilation_outward,
                                        float max_cam_to_aabox_length)
{
    WavefrontNeuralPT::this_module_setup_constants(voxel_grid_aabb,
                                                   voxel_size,
                                                   current_lod,
                                                   voxel_extent_dilation_outward,
                                                   max_cam_to_aabox_length);
}

WavefrontNeuralPT::WavefrontNeuralPT(RenderCudaBinding *backend) : backend(backend) {}

WavefrontNeuralPT::~WavefrontNeuralPT() {}

void WavefrontNeuralPT::release_mapped_display_resources() {}

bool WavefrontNeuralPT::ui_and_state()
{
    bool model_changed = false;

    return model_changed;
}

std::string WavefrontNeuralPT::name() const
{
    return "Cuda Wavefront Neural Path Tracing Extension";
}

void WavefrontNeuralPT::update_scene_from_backend(const Scene &scene) {}

void WavefrontNeuralPT::camera_raygen(clsl::NeuralBounceQuery *bounce_queries,
                                      clsl::NeuralBounceResult *bounce_results)
{
    uint32_t n_elements = m_num_rays_initialised;
    tcnn::linear_kernel(clsl::camera_raygen,
                        0,
                        0,
                        n_elements,
                        accumulated_spp,
                        sample_offset,
                        bounce_queries,
                        bounce_results);

#ifdef DEBUGGING
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
#endif
}

void WavefrontNeuralPT::accumulate_results(clsl::NeuralBounceQuery *bounce_queries,
                                           clsl::NeuralBounceResult *bounce_results)
{
    uint32_t n_elements = m_num_rays_initialised;
    tcnn::linear_kernel(clsl::accumulate_neural_results,
                        0,
                        0,
                        n_elements,
                        accumulated_spp,
                        bounce_queries,
                        bounce_results);

#ifdef DEBUGGING
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
#endif
}
