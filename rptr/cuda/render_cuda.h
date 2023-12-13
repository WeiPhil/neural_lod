#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "render_backend.h"
#include "error_io.h"

#define checkCuda(val) check_cuda_error((val), #val, __FILE__, __LINE__)

inline void check_cuda_error(cudaError_t result,
                char const *const func,
                const char *const file,
                int const line) {
    if (result)
        throw_error("CUDA Runtime error: %s\nat %s:%d \"%s\"", cudaGetErrorString(result), file, line, func);
}

namespace cuex {

struct Geometry {
    void* vertices = 0;
    void* normals = 0;
    void* uvs = 0;
    void* indices = 0;
    int vertexCount = 0;
    int triangleCount = 0;
};

struct Mesh {
    std::vector<Geometry> geometries;
};

struct ParameterizedMesh {
    void* perTriangleMaterialIDs = 0;
    int mesh_id = -1;
    bool no_alpha = false;
};

} // namespace

struct RenderCudaBinding : RenderExtension {
    RenderBackend* backend = nullptr;

    std::vector<cuex::Mesh> meshes;
    std::vector<cuex::ParameterizedMesh> parameterized_meshes;
    // note: instances as in backend
    std::vector<cudaMipmappedArray_t> textures;
    std::vector<cudaTextureObject_t> samplers;
    cudaTextureObject_t* sampler_buffer = nullptr;

    void* local_host_params = nullptr;
    void* global_params_buffer = nullptr;

    void* material_buffer = nullptr;
    void* instanced_geometry_buffer = nullptr;
    void* lights_buffer = nullptr;

    int screen_width = 0, screen_height = 0;
    cudaSurfaceObject_t accum_buffers[2] = { 0 };
    cudaSurfaceObject_t pixel_buffers[2] = { 0 };
    cudaSurfaceObject_t debug_buffer = 0;

    RenderRayQuery* ray_queries = nullptr;
    RenderRayQueryResult* ray_results = nullptr;

    static const int MAX_CONCURRENT_KERNELS = 64;
    cudaStream_t concurrent_streams[MAX_CONCURRENT_KERNELS] = { NULL };
    int num_concurrent_streams = 0;

    RenderCudaBinding(RenderBackend* backend);
    ~RenderCudaBinding();

    std::string name() const override;
    void initialize(const int fb_width, const int fb_height) override;
    void update_scene_from_backend(const Scene& scene) override;

    void create_sampler_objects(const Scene& scene);
    void release_sampler_objects();

    virtual int active_pixel_buffer_index() = 0;
    virtual int active_accum_buffer_index() = 0;
    // addressable return type for convenience in cuda update calls
    cudaSurfaceObject_t const& pixel_buffer() { return pixel_buffers[active_pixel_buffer_index()]; }
    cudaSurfaceObject_t const& accum_buffer() { return accum_buffers[active_accum_buffer_index()]; }
};

#ifdef ENABLE_VULKAN
RenderCudaBinding* create_vulkan_cuda_extension(RenderBackend* vulkan_backend);
#endif
