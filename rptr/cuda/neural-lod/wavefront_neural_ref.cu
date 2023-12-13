#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include "util/tinyexr.h"

#include <glm/gtx/rotate_vector.hpp>

#define INCLUDE_BASE_WAVEFRONT_PT
#include "../wavefront_pt.cuh"  // Wavefront PT device environment

#include "neural_lod_learning_common.cuh"
#include "wavefront_neural_ref.h"

#define CONSTANT_ENV

namespace clsl {
#define inline __device__ inline

// todo: more GLSL includes

#undef inline
}

namespace clsl {

__constant__ glm::vec3 envmap_color_const;
__constant__ glm::vec3 background_color_const;
__constant__ cudaSurfaceObject_t envmap_texture_const;

__constant__ float max_cam_to_aabox_length_const;

__device__ bool bounce_ray_constant_ref(int i,
                                        int bounce,
                                        WavefrontBounceData const &data,
                                        bool apply_rr,
                                        bool show_visibility_map,
                                        bool use_envmap,
                                        float envmap_rotation)
{
    // unpack hit
    vec3 ray_origin = data.bounce_queries[i].origin;
    vec3 ray_dir = data.bounce_queries[i].dir;
    RenderRayQueryResult ray_result = data.bounce_results[i];

    float approx_tri_solid_angle;
    RTHit hit = ray_hit(ray_origin, ray_dir, ray_result.result, approx_tri_solid_angle);

    // unpack state
    BounceState state = unpack_bounce(i, bounce, data);

#ifdef STATS_NUM_BOUNCES
    if (view_params.rparams.output_moment == 1) {
        state.illum = vec3(bounce);
    }
#endif

    if (!(hit.dist > 0.0f)) {
        // state.illum += bounce == 0 ? clsl::background_color_const : state.path_throughput *
        // clsl::envmap_color_const;
#ifdef STATS_NUM_BOUNCES
        if (view_params.rparams.output_moment != 1)
#endif
            if (use_envmap) {
                glm::vec3 dir = ray_dir;
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
                state.illum = bounce == 0 ? envmap_val : state.path_throughput * envmap_val;
            } else {
                state.illum = bounce == 0 ? clsl::background_color_const
                                          : state.path_throughput * clsl::envmap_color_const;
            }

        return pack_bounce(i, bounce, data, state, false);
    }

    mat2 duvdxy = mat2(0.0f);
    HitPoint surface_lookup_point = HitPoint{ray_origin + hit.dist * ray_dir, hit.uv, duvdxy};

    if (show_visibility_map) {
        float depth_estimate = length(view_params.cam_pos - surface_lookup_point.p) /
                               clsl::max_cam_to_aabox_length_const;

        state.illum = vec3(1.0f) * (1.f - max(depth_estimate, 0.0f));
        return pack_bounce(i, bounce, data, state, false);
    }

    MATERIAL_TYPE mat;
    EmitterParams emit;
    float material_alpha = unpack_material(
        mat, emit, hit.material_id, materials[hit.material_id], surface_lookup_point);

    // For opaque objects (or in the future, thin ones) make the normal face forward
    bool flip_forward = !(mat.specular_transmission > 0.f);
    InteractionPoint interaction =
        surface_interaction(surface_lookup_point.p, hit, ray_dir, flip_forward);

    bool recurse = material_scattering(-ray_dir,
                                       interaction,
                                       state.path_throughput,
                                       material_alpha,
                                       i,
                                       bounce,
                                       data,
                                       mat,
                                       state.rng,
                                       state.prev_bsdf_pdf);

    if (apply_rr)
        recurse = recurse && rr_terminate(bounce, state.path_throughput, state.rng);

    return pack_bounce(i, bounce, data, state, recurse);
}

__global__ void bounce_ray_constant_ref(int bounce,
                                        WavefrontBounceData data,
                                        bool apply_rr,
                                        bool show_visibility_map,
                                        bool use_envmap,
                                        float envmap_rotation)
{
    int i = WavefrontPixelIndex;
    if (i >= fb_width * fb_height)
        return;

    bool path_active = data.bounce_queries[i].mode_or_data >= 0;
    if (path_active) {
        path_active = bounce_ray_constant_ref(
            i, bounce, data, apply_rr, show_visibility_map, use_envmap, envmap_rotation);
        if (!path_active)
            data.bounce_queries[i].mode_or_data = -1;
    }
}

}  // namespace

WavefrontConstantRef::WavefrontConstantRef(RenderCudaBinding *backend) : WavefrontPT(backend)
{
}

WavefrontConstantRef::~WavefrontConstantRef() {}

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

void WavefrontConstantRef::load_envmap()
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

void WavefrontConstantRef::initialize(const int fb_width, const int fb_height)
{
    this->WavefrontPT::initialize(fb_width, fb_height);
}

void WavefrontConstantRef::setup_constants()
{
    this->WavefrontPT::setup_constants();
    this->WavefrontPT::this_module_setup_constants();

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

void WavefrontConstantRef::bounce_rays(int bounce)
{
    clsl::WavefrontBounceData data = bounce_data(bounce);

    dim3 dimBlock(32, 16);
    dim3 dimGrid((backend->screen_width + dimBlock.x - 1) / dimBlock.x,
                 (backend->screen_height + dimBlock.y - 1) / dimBlock.y);
    clsl::bounce_ray_constant_ref<<<dimGrid, dimBlock>>>(bounce,
                                                         data,
                                                         params.apply_rr,
                                                         params.show_visibility_map,
                                                         params.use_envmap,
                                                         params.envmap_rotation);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void WavefrontConstantRef::render(bool reset_accumulation)
{
    setup_constants();

    // Pre-compute approximate max distance for depth map
    {
        float max_cam_to_aabox_length = -1.f;
        auto extent = voxel_grid_aabb.extent();
        for (size_t a = 0; a < 2; a++)
            for (size_t b = 0; b < 2; b++)
                for (size_t c = 0; c < 2; c++) {
                    float lenght =
                        glm::length(backend->backend->camera.pos -
                                    glm::vec3(voxel_grid_aabb.min[0] + a * extent[0],
                                              voxel_grid_aabb.min[1] + b * extent[1],
                                              voxel_grid_aabb.min[2] + c * extent[2]));
                    max_cam_to_aabox_length = max(lenght, max_cam_to_aabox_length);
                }
        checkCuda(cudaMemcpyToSymbol(
            clsl::max_cam_to_aabox_length_const, &max_cam_to_aabox_length, sizeof(float)));
    }

    if (reset_accumulation) {
        sample_offset += accumulated_spp;
        accumulated_spp = 0;
    }

    for (int spp = 0; spp < backend->backend->params.batch_spp; ++spp) {
        camera_raygen();

        for (int bounce = 0; bounce < params.max_depth; ++bounce) {
            bool shadow_rays = false;

            cast_rays(bounce, shadow_rays);

            bounce_rays(bounce);
        }
        accumulate_results(spp);
    }
}