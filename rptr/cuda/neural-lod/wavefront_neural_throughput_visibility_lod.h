#pragma once

#include "../render_cuda.h"
#include "../wavefront_neural_pt.h"
#include <glm/glm.hpp>

#ifdef TCNN_NAMESPACE_BEGIN
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#endif

#include "neural_lod_learning.h"

#ifdef TCNN_NAMESPACE_BEGIN
using tcnn_network = tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>;
#endif


struct WavefrontNeuralThroughputVisibilityLodParams {
    uint32_t max_depth = 3;
    bool apply_rr = false;
    uint32_t rr_start_bounce = 1;
    float inference_min_extent_dilation = 1e-4;
    bool stochastic_threshold = false;
    bool use_hg = false;
    float hg_g = 0.5f;
    glm::vec3 envmap_color = glm::vec3(1.0f);
    glm::vec3 background_color = glm::vec3(1.0f);
    int current_lod = 0;
    bool apply_visibility_rr = false;
    bool use_envmap = false;
    float envmap_rotation = 0.f;
    char envmap_filename[MAX_FILENAME_SIZE] = "";
};

struct WavefrontNeuralThroughputVisibilityLod : WavefrontNeuralPT {


#ifdef TCNN_NAMESPACE_BEGIN
    struct NeuralData {
        // Pre-allocated during initialisation for at most the screen size 
        float* visibility_network_inputs;
        float* visibility_network_outputs;
        float* throughput_network_inputs;
        float* throughput_network_outputs;

        tcnn::GPUMemory<uint32_t> active_counter;
        tcnn::GPUMemory<uint32_t> active_intersection_counter;
    };

#else
    struct NeuralData;
#endif

#ifdef TCNN_NAMESPACE_BEGIN
    struct EnvmapData {
        tcnn::GPUMemory<float> envmap_image;
    };

#else
    struct EnvmapData;
#endif

    std::unique_ptr<NeuralData> nn;
    std::unique_ptr<EnvmapData> envmap_data;
    
    cudaTextureObject_t envmap_texture = 0;

    WavefrontNeuralThroughputVisibilityLodParams params;

    WavefrontNeuralThroughputVisibilityLod(RenderCudaBinding* backend);
    ~WavefrontNeuralThroughputVisibilityLod();

    void load_envmap();

    bool ui_and_state() override;
    std::string name() const override;
    //char const* const* variant_names() override;

    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    void render(bool reset_accumulation) override;

    void render(bool reset_accumulation,
                neural_lod_learning::NeuralLodLearning::VisibilityNN *visibility_neural_net,
                neural_lod_learning::NeuralLodLearning::ThroughputNN *throughput_neural_net,
                bool throughput_log_learning,
                neural_lod_learning::NeuralLodLearning::VoxelGrid* voxel_grid,
                int current_lod,
                float voxel_extent_dilation_outward);

    void setup_constants(neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
                         float voxel_size,
                         int current_lod,
                         float voxel_extent_dilation_outward,
                         float max_cam_to_aabox_length) override;

#ifdef TCNN_NAMESPACE_BEGIN
    void voxel_tracing(int sample_idx,
                       tcnn_network *visibility_network,
                       tcnn_network *throughput_network,
                       bool throughput_log_learning,
                       const nanovdb::FloatGrid **d_grids);
#endif

    void release_mapped_display_resources() override;
};
