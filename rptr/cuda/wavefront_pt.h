#pragma once

#include "render_cuda.h"
#include <glm/glm.hpp>

// #define STATS_NUM_BOUNCES
#define STATS_VARIANCE

#define WavefrontPixelIndex (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))
#define WavefrontPixelID ( (glm::uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y) & ~glm::uvec2(0x3 << 3, 0x3)) \
    + (glm::uvec2((blockIdx.y * blockDim.y + threadIdx.y) << 3, (blockIdx.x * blockDim.x + threadIdx.x) >> 3) & glm::uvec2(0x3 << 3, 0x3)) )

struct WavefrontPTParams {
    
};

namespace clsl {
    struct WavefrontBounceData;
}

struct WavefrontPT : RenderExtension {
    RenderCudaBinding* backend;
    int rt_rayquery_index = -1;

    glm::vec4* throughtput_pdf = nullptr;
    glm::vec4* illumination_rnd = nullptr;
    glm::vec4* nee_pdf = nullptr;
    int* real_bounce = nullptr;

    WavefrontPTParams params;
    int accumulated_spp = 0;
    int sample_offset = 0;

    WavefrontPT(RenderCudaBinding* backend);
    ~WavefrontPT();

    bool ui_and_state(bool& renderer_changed) override;
    std::string name() const override;
    //char const* const* variant_names() override;

    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    virtual void render(bool reset_accumulation);

    inline void this_module_setup_constants(); // must be inline due to constant linking constraints
    virtual void setup_constants();
    clsl::WavefrontBounceData bounce_data(int bounce);

    virtual void camera_raygen();
    virtual void cast_rays(int bounce, bool shadow_rays);
    virtual void bounce_rays(int bounce);
    virtual void postprocess_bounce(int bounce);
    virtual void accumulate_results(int sample_idx);

    void release_mapped_display_resources() override;
};
