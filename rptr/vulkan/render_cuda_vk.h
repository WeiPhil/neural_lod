#pragma once

#include "render_vulkan.h"
#include "../cuda/render_cuda.h"

struct RenderVulkanCuda : RenderCudaBinding {
    int cudaDeviceId;

    std::vector<cudaExternalMemory_t> persistent_external_memory;
    std::vector<cudaExternalMemory_t> scene_external_memory;
    std::vector<cudaExternalMemory_t> display_external_memory;
    std::vector<cudaMipmappedArray_t> display_textures;
    std::vector<cudaMipmappedArray_t> debug_textures;
    std::vector<void*> scene_external_pointers;

    RenderVulkanCuda(RenderVulkan* backend);
    virtual ~RenderVulkanCuda();

    std::string name() const override;
    RenderVulkan* vulkan_backend() const { return static_cast<RenderVulkan*>(this->backend); }
    
    void initialize(const int fb_width, const int fb_height) override;

    void update_scene_from_backend(const Scene& scene) override;

    void release_mapped_resources();
    void release_mapped_scene_resources(const Scene* release_changes_only) override;
    void release_mapped_display_resources() override;

    virtual int active_pixel_buffer_index() override;
    virtual int active_accum_buffer_index() override;
};
