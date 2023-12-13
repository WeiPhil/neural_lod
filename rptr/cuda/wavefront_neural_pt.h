#pragma once

#include "render_cuda.h"
#include <glm/glm.hpp>

#ifdef TCNN_NAMESPACE_BEGIN
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common.h>
#endif

#include "neural-lod/neural_lod_learning.h"

// #define STATS_NUM_BOUNCES
#define STATS_VARIANCE

#define WavefrontPixelIndex (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))
#define WavefrontPixelID ( (glm::uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y) & ~glm::uvec2(0x3 << 3, 0x3)) \
    + (glm::uvec2((blockIdx.y * blockDim.y + threadIdx.y) << 3, (blockIdx.x * blockDim.x + threadIdx.x) >> 3) & glm::uvec2(0x3 << 3, 0x3)) )


struct WavefrontNeuralPTParams {
    
};

#define INACTIVE_PATH -3
#define ESCAPED_PATH -2
#define ACTIVE_PATH -1
#define ACTIVE_PATH_WITH_INTERSECTION 0

// Scene & Includes
namespace clsl {
    using namespace glm;
    #define inline __device__ inline

    #include "../rendering/language.hpp"
    #include "../rendering/pointsets/lcg_rng.glsl"
    
    #undef inline
} // namespace

namespace clsl {

    struct NeuralBounceQuery {
    // Queries
        glm::vec3 origin;
        int path_state;
        glm::vec3 dir;
        LCGRand rng;
    };

    struct NeuralBounceResult {
        glm::ivec3 lod_voxel_idx;
        float t_voxel_near;
        glm::vec3 path_throughput;
        float t_voxel_far;  
        glm::vec3 throughput_wi;
        float throughput_inference_pdf_inv;
        glm::vec3 illumination;
        uint32_t pixel_coord;
        float voxel_threshold;  
        uint32_t needed_inferences;
    };

    struct WavefrontNeuralBounceData {

        void set(NeuralBounceQuery* queries, NeuralBounceResult* results, size_t size) {
            this->queries = queries;
            this->results = results;
            this->size = size;
        }

        NeuralBounceQuery *queries;
        NeuralBounceResult *results;
        size_t size;
    };
}

struct WavefrontNeuralPT : RenderExtension {
    RenderCudaBinding* backend;
    int rt_rayquery_index = -1;

#ifdef TCNN_NAMESPACE_BEGIN
    struct TcnnWrapper {
        tcnn::GPUMemoryArena::Allocation scratch_alloc;
    };
#else
    struct TcnnWrapper;
#endif

    std::unique_ptr<TcnnWrapper> m_tcnn_wrapper;
    uint32_t m_num_rays_initialised;

    clsl::WavefrontNeuralBounceData m_wavefront_neural_data[2];
    clsl::WavefrontNeuralBounceData m_wavefront_neural_data_intersected;
    
    float* m_visibility_network_inputs;
    float* m_visibility_network_outputs;
    float* m_throughput_network_inputs;
    float* m_throughput_network_outputs;

    WavefrontNeuralPTParams params;
    int accumulated_spp = 0;
    int sample_offset = 0;

    WavefrontNeuralPT(RenderCudaBinding* backend);
    ~WavefrontNeuralPT();

    bool ui_and_state(bool& renderer_changed) override { return ui_and_state(); } // renderer and model changes tracked manually
    virtual bool ui_and_state();
    std::string name() const override;

    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    virtual void render(bool reset_accumulation) = 0;

    inline void this_module_setup_constants(
        neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
        float voxel_size,
        int current_lod,
        float voxel_extent_dilation_outward,
        float max_cam_to_aabox_length);  // must be inline due to constant linking constraints

    virtual void setup_constants(neural_lod_learning::NeuralLodLearning::AABox<glm::vec3> voxel_grid_aabb,
                                 float voxel_size,
                                 int current_lod,
                                 float voxel_extent_dilation_outward,
                                 float max_cam_to_aabox_length);

    void camera_raygen(clsl::NeuralBounceQuery *bounce_queries,
                       clsl::NeuralBounceResult *bounce_results);

    void accumulate_results(clsl::NeuralBounceQuery *bounce_queries,
                            clsl::NeuralBounceResult *bounce_results);

    void release_mapped_display_resources() override;
};
