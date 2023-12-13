#pragma once

#include "../render_cuda.h"
#include "../wavefront_pt.h"
#include <glm/glm.hpp>

#include "neural_lod_learning.h"

#ifdef TCNN_NAMESPACE_BEGIN
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_memory.h>
#endif

#ifdef TCNN_NAMESPACE_BEGIN
using tcnn_network = tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>;
#endif


struct WavefrontConstantRefParams {
    glm::vec3 envmap_color = glm::vec3(1.0f);
    glm::vec3 background_color = glm::vec3(1.0f);
    int max_depth = 9;
    bool apply_rr = false;
    bool show_visibility_map = false;
    bool use_envmap = false;
    float envmap_rotation = 0.f;
    char envmap_filename[MAX_FILENAME_SIZE] = "";
};

struct WavefrontConstantRef : WavefrontPT {

#ifdef TCNN_NAMESPACE_BEGIN
    struct EnvmapData {
        tcnn::GPUMemory<float> envmap_image;
    };

#else
    struct EnvmapData;
#endif

    std::unique_ptr<EnvmapData> envmap_data;
    cudaTextureObject_t envmap_texture = 0;

    WavefrontConstantRefParams params;

    // Scene bounding box info
    neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> scene_aabb = {
        glm::vec3(2e32f), glm::vec3(-2e32f)};
    neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb = {
        glm::vec3(2e32f), glm::vec3(-2e32f)};

    WavefrontConstantRef(RenderCudaBinding* backend);
    ~WavefrontConstantRef();

    void load_envmap();

    bool ui_and_state(bool& renderer_changed) override;
    std::string name() const override;
    //char const* const* variant_names() override;

    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    void render(bool reset_accumulation) override;

    void setup_constants() override;

    void bounce_rays(int bounce) override;

    void release_mapped_display_resources() override;
};
