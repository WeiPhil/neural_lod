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


struct WavefrontNeuralVisibilityLodParams {
    uint32_t max_visibility_inferences = 100;
    bool apply_rr = false;
    float inference_min_extent_dilation = 1e-4;
    bool stochastic_threshold = false;
    bool display_needed_inferences = false;
    glm::vec3 visibility_color = glm::vec3(1.0f);
    glm::vec3 background_color = glm::vec3(1.0f);
    int current_lod = 0;
};

struct WavefrontNeuralVisibilityLod : WavefrontNeuralPT {


#ifdef TCNN_NAMESPACE_BEGIN
    struct NeuralData {
        // Pre-allocated during initialisation for at most the screen size 
        float* visibility_network_inputs;
        float* visibility_network_outputs;

        tcnn::GPUMemory<uint32_t> active_counter;
		tcnn::GPUMemory<uint32_t> final_counter;
    };

#else
    struct NeuralData;
#endif

    std::unique_ptr<NeuralData> nn;

    WavefrontNeuralVisibilityLodParams params;

    WavefrontNeuralVisibilityLod(RenderCudaBinding* backend);
    ~WavefrontNeuralVisibilityLod();

    bool ui_and_state() override;
    std::string name() const override;
    //char const* const* variant_names() override;

    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    void render(bool reset_accumulation) override;

    void render(bool reset_accumulation,
                neural_lod_learning::NeuralLodLearning::VisibilityNN *visibility_neural_net,
                neural_lod_learning::NeuralLodLearning::VoxelGrid* voxel_grid,
                int current_lod,
                float voxel_extent_dilation_outward);

    void setup_constants(neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
                         float voxel_size,
                         int current_lod,
                         float voxel_extent_dilation_outward,
                         float max_cam_to_aabox_length) override;

#ifdef TCNN_NAMESPACE_BEGIN
    void voxel_tracing(tcnn_network *visibility_network,
                       const nanovdb::FloatGrid **d_grids);
#endif

    void release_mapped_display_resources() override;
};
