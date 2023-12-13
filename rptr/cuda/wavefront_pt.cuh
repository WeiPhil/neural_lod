#include <cuda_runtime_api.h>
#define GLCPP_DEFAULT(x) // avoid dynamic initialization in CUDA constant vars
#ifdef INCLUDE_BASE_WAVEFRONT_PT
#include "wavefront_pt.h"
#endif

#define CAST_SHADOW_RAYS

// Framebuffer
namespace clsl {

__constant__ cudaSurfaceObject_t framebuffer;
__constant__ cudaSurfaceObject_t accum_buffer;
__constant__ int fb_width;
__constant__ int fb_height;

} // namespace

// Scene & Includes
namespace clsl {
    using namespace glm;
    #define inline __device__ inline

    #include "../rendering/language.hpp"
    #include "../rendering/pointsets/lcg_rng.glsl"
    #include "../rendering/util.glsl"
    
    #include "../vulkan/gpu_params.glsl"
    #define DEFAULT_GEOMETRY_BUFFER_TYPES
    #include "../rendering/rt/geometry.h.glsl"
    #include "../rendering/rt/hit.glsl"

    #include "../rendering/pathspace.h"

    #define MATERIAL_DECODE_CUSTOM_TEXTURES
    #define GLTF_SUPPORT_TRANSMISSION
    #include "../rendering/rt/materials.glsl"
    #include "../rendering/bsdfs/gltf_bsdf.glsl"

    #include "../rendering/lights/tri.glsl"

    __constant__ ViewParams view_params;
    __constant__ RenderParams render_params;
    __constant__ SceneParams scene_params;

    __constant__ InstancedGeometry* instances;
    __constant__ MATERIAL_PARAMS* materials;
    __constant__ TriLightData* lights;
    __constant__ cudaTextureObject_t* textures;

    #include "../rendering/rt/material_textures.glsl"

    #define SCENE_GET_LIGHT_SOURCE(light_id) decode_tri_light(lights[light_id])
    #define SCENE_GET_LIGHT_SOURCE_COUNT()   int(scene_params.light_sampling.light_count)

    #define BINNED_LIGHTS_BIN_SIZE int(view_params.light_sampling.bin_size)
    #define SCENE_GET_BINNED_LIGHTS_BIN_COUNT() (int(scene_params.light_sampling.light_count + (view_params.light_sampling.bin_size - 1)) / int(view_params.light_sampling.bin_size))

    #include "../rendering/mc/nee.glsl"

    #undef inline
} // namespace

// Path tracing
namespace clsl {

struct WavefrontBounceData {
    RenderRayQuery *bounce_queries;
    RenderRayQueryResult *bounce_results;
    glm::vec4 *throughtput_pdf;
    glm::vec4 *illumination_rnd;
    RenderRayQuery *shadow_queries;
    RenderRayQueryResult *shadow_results;
    glm::vec4 *nee_pdf;
};

struct BounceState {
    vec3 illum;
    RANDOM_STATE rng;
    vec3 path_throughput;
    float prev_bsdf_pdf;
};

__device__ inline void camera_raygen(int i, float xf, float yf, uint32_t rnd_state
    , RenderRayQuery *queries, glm::vec4 *throughtput_pdf, glm::vec4 *illumination_rnd) {
    queries[i].origin = vec3(view_params.cam_pos);
    queries[i].dir = normalize(xf * vec3(view_params.cam_du) + yf * vec3(view_params.cam_dv) + vec3(view_params.cam_dir_top_left));
    queries[i].t_max = 2.e32f;
    queries[i].mode_or_data = 0;

    throughtput_pdf[i] = vec4(1.0f, 1.0f, 1.0f, 2.e16f);
    illumination_rnd[i] = vec4(0.0f, 0.0f, 0.0f, (float const&) rnd_state);
}

__device__ inline BounceState unpack_bounce(int i, int bounce, WavefrontBounceData const& data) {
    BounceState state;

    vec4 illum_rnd = data.illumination_rnd[i];
    vec4 throughput_last_pdf = data.throughtput_pdf[i];

    state.rng = { floatBitsToUint(illum_rnd.w) };
    state.illum = vec3(illum_rnd);
    state.path_throughput = vec3(throughput_last_pdf);
    state.prev_bsdf_pdf = throughput_last_pdf.w;
    return state;
}

__device__ inline bool pack_bounce(int i, int bounce, WavefrontBounceData const& data
    , BounceState const& state, bool recurse) {
    if (recurse)
        data.throughtput_pdf[i] = vec4(state.path_throughput, state.prev_bsdf_pdf);
    data.illumination_rnd[i] = vec4(state.illum, uintBitsToFloat(state.rng.state));

    return recurse;
}

__device__ inline RTHit ray_hit(vec3 ray_origin, vec3 ray_dir, vec4 ray_result, float& approx_tri_solid_angle) {
    RTHit hit;
    vec2 attrib = vec2(ray_result);
    int instancedGeometryIdx = floatBitsToInt(ray_result.z);
    int primitiveIdx = floatBitsToInt(ray_result.w);

    // miss
    if (instancedGeometryIdx == -1) {
        hit.dist = -1.f;
        return hit;
    }

    InstancedGeometry instance = instances[instancedGeometryIdx];
    RenderMeshParams geom = instance.geometry;

    const uvec3 idx = (geom.indices.i) ? geom.indices.i[primitiveIdx] : uvec3(primitiveIdx * 3) + uvec3(0, 1, 2);

    mat3 vertices = calc_hit_vertices(geom.vertices,
#ifdef QUANTIZED_POSITIONS
        geom.quantized_scaling, geom.quantized_offset,
#endif
    idx);
    vec3 hit_p = vertices * vec3(1.0f - attrib.x - attrib.y, attrib.x, attrib.y);
    hit_p = vec3(instance.instance_to_world * vec4(hit_p, 1.0f));
    hit = calc_hit_attributes(length(hit_p - ray_origin), primitiveIdx, attrib,
        vertices, idx, transpose(mat3(instance.world_to_instance)),
        geom.normals, geom.num_normals > 0,
#ifndef QUANTIZED_NORMALS_AND_UVS
        geom.uvs,
#endif
        geom.num_uvs > 0,
        geom.material_id, geom.materials
        );

    approx_tri_solid_angle = length(hit.geo_normal);
    hit.geo_normal /= approx_tri_solid_angle;
    approx_tri_solid_angle *= abs(dot(hit.geo_normal, ray_dir)) / (hit.dist * hit.dist);

    return hit;
}

__device__ inline InteractionPoint surface_interaction(vec3 hit_pos, RTHit const& hit
    , vec3 ray_dir, bool flip_forward) {
    InteractionPoint interaction;
    interaction.p = hit_pos;
    interaction.gn = hit.geo_normal;
    interaction.n = hit.normal;

    vec3 w_o = -ray_dir;
    // For opaque objects make the normal face forward
    if (flip_forward && dot(w_o, interaction.gn) < 0.0) {
        interaction.n = -interaction.n;
        interaction.gn = -interaction.gn;
    }

    // apply normal mapping
    int normal_map = materials[hit.material_id].normal_map;
    if (normal_map != -1) {
        vec3 v_y = normalize( cross(hit.normal, hit.tangent) );
        vec3 v_x = cross(v_y, hit.normal);
        v_x *= length(hit.tangent);
        v_y *= hit.bitangent_l;

        vec3 map_nrm = vec3(*(vec4 const*) &tex2D<float4>(textures[normal_map], hit.uv.x, hit.uv.y).x);
        map_nrm = vec3(2.0f, 2.0f, 1.0f) * map_nrm - vec3(1.0f, 1.0f, 0.0f);
        // Z encoding might be unclear, just reconstruct
        map_nrm.z = sqrt(max(1.0f - map_nrm.x * map_nrm.x - map_nrm.y * map_nrm.y, 0.0f));
        mat3 iT_shframe = mat3(v_x, v_y, scene_params.normal_z_scale * interaction.n);
        interaction.n = normalize(iT_shframe * map_nrm);
    }

    // fix incident directions under geo hemisphere
    // note: this may violate strict energy conservation, due to scattering into more than the hemisphere
    {
        float nw = dot(w_o, interaction.n);
        float gnw = dot(w_o, interaction.gn);
        if (nw * gnw <= 0.0f) {
            float blend = gnw / (gnw - nw);
            //   dot(blend * interaction.n + (1-blend) * interaction.gn, w_o)
            // = (nw * gnw + (gnw - nw) * gnw - gnw * gnw) / (gnw - nw)
            // = 0
            interaction.n = normalize( mix(interaction.gn, interaction.n, blend - EPSILON) );
        }
    }
    
    interaction.v_y = normalize( cross(interaction.n, hit.tangent) );
    interaction.v_x = cross(interaction.v_y, interaction.n);

    return interaction;
}

__device__ inline vec3 local_illumination(vec3 w_o, InteractionPoint const& interaction
    , vec3 scatter_throughput, float prev_bsdf_pdf
    , int i, int bounce, WavefrontBounceData const& data
    , MATERIAL_TYPE const& mat
    , EmitterParams const& emit, float approx_tri_solid_angle
    , RANDOM_STATE& rng) {
    vec3 illum = vec3(0.0f);

    // direct emitter hit
    if (render_params.output_channel == 0 && emit.radiance != vec3(0.0f))
    {
        float light_pdf;
        if (view_params.light_sampling.light_mis_angle > 0.0f)
            light_pdf = 1.0f / view_params.light_sampling.light_mis_angle;
        else
            light_pdf = wpdf_direct_tri_light(approx_tri_solid_angle);
        float w = nee_mis_heuristic(1.f, prev_bsdf_pdf, 1.f, light_pdf);
        illum += w * scatter_throughput * emit.radiance;
    }

    // NEE
    vec4 nee_contrib_pdf = vec4(0.0f);
    if (render_params.output_channel == 0 && bounce+1 < render_params.max_path_depth) {
        // first two dimensions light position selection, last light selection (sky/direct)
        vec4 nee_rng_sample = vec4(RANDOM_FLOAT2(rng, DIM_POSITION_X), RANDOM_FLOAT2(rng, DIM_LIGHT_SEL_1));
        NEEQueryAux nee_aux;
        nee_aux.mis_pdf = view_params.light_sampling.light_mis_angle > 0.0f ? 1.0f / view_params.light_sampling.light_mis_angle : 0.0f;
        vec3 nee_contrib = scatter_throughput * sample_direct_light(mat, interaction, w_o, vec2(nee_rng_sample), *(vec2 const*) &nee_rng_sample.z, nee_aux);
#ifdef CAST_SHADOW_RAYS
        nee_contrib_pdf = vec4(nee_contrib, nee_aux.mis_pdf);
        bool cast_shadow_ray = nee_aux.mis_pdf > 0.0f;
        if (cast_shadow_ray) {
            data.shadow_queries[i].origin = interaction.p;
            data.shadow_queries[i].mode_or_data = 0;
            data.shadow_queries[i].dir = nee_aux.light_dir;
            data.shadow_queries[i].t_max = nee_aux.light_dist; // todo: epsilon?
        }
#else
        illum += nee_contrib;
#endif
    }
    data.nee_pdf[i] = nee_contrib_pdf;

    // AOVs
    if (render_params.output_channel != 0) {
        float reliability = pow(0.25f, (float) bounce);
        if (render_params.output_channel == 1)
            illum += scatter_throughput * mat.base_color * reliability;
        else if (render_params.output_channel == 2) {
            illum += interaction.n * reliability;
        }
        else if (render_params.output_channel == 3) {
            illum += interaction.p * reliability;
        }
    }

    return illum;
}

__device__ inline bool material_scattering(vec3 w_o, InteractionPoint const& interaction
    , vec3& path_throughput, float material_alpha
    , int i, int bounce, WavefrontBounceData const& data
    , MATERIAL_TYPE const& mat, RANDOM_STATE& rng, float& prev_bsdf_pdf) {
    vec3 ray_dir;
    if (material_alpha < 1.0f && (0.0f >= material_alpha || RANDOM_FLOAT1(rng, DIM_FREE_PATH) >= material_alpha)) {
        // transparent
        ray_dir = -w_o;
    }
    else {
        vec2 bsdfLobeSample = RANDOM_FLOAT2(rng, DIM_LOBE);
        vec2 bsdfDirSample = RANDOM_FLOAT2(rng, DIM_DIRECTION_X);

        vec3 w_i;
        float sampling_pdf, mis_wpdf;
        vec3 bsdf = sample_bsdf(mat, interaction, w_o, w_i, sampling_pdf, mis_wpdf, bsdfDirSample, bsdfLobeSample, rng);
        if (mis_wpdf == 0.f || bsdf == vec3(0.f) || !(dot(w_i, interaction.n) * dot(w_i, interaction.gn) > 0.0f))
            return false;

        path_throughput *= bsdf;

        prev_bsdf_pdf = mis_wpdf;
        ray_dir = w_i;
    }

    data.bounce_queries[i].origin = interaction.p;
    data.bounce_queries[i].dir = ray_dir;

    return true;
}

__device__ inline bool rr_terminate(int bounce, vec3& path_throughput, RANDOM_STATE& rng) {
    // Russian roulette termination
    if (bounce > 1) {
        float prefix_weight = min(max(path_throughput.x, max(path_throughput.y, path_throughput.z)), 1.0f);

        float rr_prob = prefix_weight;
        float rr_sample = RANDOM_FLOAT1(rng, DIM_RR);
        if (bounce > 6)
            rr_prob = min(0.95f, rr_prob); // todo: good?

        if (rr_sample < rr_prob)
            path_throughput /= rr_prob;
        else
            return false;
    }

    return true;
}

__device__ inline bool bounce_ray(int i, int bounce, WavefrontBounceData const& data) {
    // unpack hit
    vec3 ray_origin = data.bounce_queries[i].origin;
    vec3 ray_dir = data.bounce_queries[i].dir;
    RenderRayQueryResult ray_result = data.bounce_results[i];

    float approx_tri_solid_angle;
    RTHit hit = ray_hit(ray_origin, ray_dir, ray_result.result, approx_tri_solid_angle);
    if (!(hit.dist > 0.0f))
        return false;

    mat2 duvdxy = mat2(0.0f);
    HitPoint surface_lookup_point = HitPoint{ray_origin + hit.dist * ray_dir, hit.uv, duvdxy};

    // unpack state
    BounceState state = unpack_bounce(i, bounce, data);
    RANDOM_SET_DIM(state.rng, DIM_CAMERA_END + bounce * (DIM_VERTEX_END + DIM_LIGHT_END));

    MATERIAL_TYPE mat;
    EmitterParams emit;
    float material_alpha = unpack_material(mat, emit
        , hit.material_id, materials[hit.material_id]
        , surface_lookup_point);
    vec3 scatter_throughput = state.path_throughput * material_alpha;

    // For opaque objects (or in the future, thin ones) make the normal face forward
    bool flip_forward = !(mat.specular_transmission > 0.f);
    InteractionPoint interaction = surface_interaction(surface_lookup_point.p, hit
        , ray_dir, flip_forward);

    state.illum += local_illumination(-ray_dir, interaction
        , scatter_throughput, state.prev_bsdf_pdf
        , i, bounce, data
        , mat, emit, approx_tri_solid_angle, state.rng);

    RANDOM_SHIFT_DIM(state.rng, DIM_LIGHT_END);

    bool recurse = material_scattering(-ray_dir, interaction
        , state.path_throughput, material_alpha
        , i, bounce, data
        , mat, state.rng, state.prev_bsdf_pdf);

    RANDOM_SHIFT_DIM(state.rng, DIM_VERTEX_END-1); // todo: potentially reusing rand, not great

    recurse = recurse && rr_terminate(bounce, state.path_throughput, state.rng);
    RANDOM_SHIFT_DIM(state.rng, 1);

    
    return pack_bounce(i, bounce, data, state, recurse);
}

__device__ inline void postprocess_bounce(int i, int bounce, glm::vec4 *illumination_rnd
    , glm::vec4* nee_pdf, RenderRayQueryResult* shadow_results) {

#ifdef CAST_SHADOW_RAYS
    vec4 nee_contrib_pdf = nee_pdf[i];
    if (nee_contrib_pdf.w > 0.0f && floatBitsToInt(shadow_results[i].result.z) == -1) {
        vec4 illum_rnd = illumination_rnd[i];
        vec3 illum = vec3(illum_rnd);
        illum += vec3(nee_contrib_pdf);
        illumination_rnd[i] = vec4(illum, illum_rnd.w);
    }
#endif
}

} // namespace

// Utils / required functionality
namespace clsl {

__device__ inline vec4 textured_color_param(const vec4 x, GLSL_in(HitPoint) hit) {
    const uint32_t mask = floatBitsToUint(x.x);
    if (IS_TEXTURED_PARAM(mask) != 0) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        return *(vec4 const*) &tex2D<float4>(textures[tex_id], hit.uv.x, hit.uv.y).x;
    }
    return x;
}
__device__ inline float textured_scalar_param(const float x, GLSL_in(HitPoint) hit) {
    const uint32_t mask = floatBitsToUint(x);
    if (IS_TEXTURED_PARAM(mask) != 0) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return (&tex2D<float4>(textures[tex_id], hit.uv.x, hit.uv.y).x)[channel];
    }
    return x;
}

__device__ inline bool raytrace_test_visibility(const vec3 from, const vec3 dir, float dist) {
    return true; // needs to be handled by outside logic for Wavefront PT
}

} // namespace

#ifdef INCLUDE_BASE_WAVEFRONT_PT
__forceinline__ void WavefrontPT::this_module_setup_constants() {
    checkCuda(cudaMemcpyToSymbol(clsl::framebuffer, &backend->pixel_buffer(), sizeof(clsl::framebuffer)));
    checkCuda(cudaMemcpyToSymbol(clsl::accum_buffer, &backend->accum_buffer(), sizeof(clsl::accum_buffer)));
    checkCuda(cudaMemcpyToSymbol(clsl::fb_width, &backend->screen_width, sizeof(clsl::fb_width)));
    checkCuda(cudaMemcpyToSymbol(clsl::fb_height, &backend->screen_height, sizeof(clsl::fb_height)));

    auto global_params_buffer = (clsl::GlobalParams*) backend->global_params_buffer;
    auto local_host_params = (clsl::LocalParams*) backend->local_host_params;
    //checkCuda(cudaMemcpyToSymbol(clsl::view_params, &global_params_buffer->view_params, sizeof(clsl::view_params), 0, cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpyToSymbol(clsl::view_params, &local_host_params->view_params, sizeof(clsl::view_params), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(clsl::scene_params, &global_params_buffer->scene_params, sizeof(clsl::scene_params), 0, cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpyToSymbol(clsl::render_params, &global_params_buffer->render_params, sizeof(clsl::render_params), 0, cudaMemcpyDeviceToDevice));
    //checkCuda(cudaMemcpyToSymbol(clsl::render_params, &local_host_params->render_params, sizeof(clsl::render_params), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpyToSymbol(clsl::instances, &backend->instanced_geometry_buffer, sizeof(clsl::instances)));
    checkCuda(cudaMemcpyToSymbol(clsl::materials, &backend->material_buffer, sizeof(clsl::materials)));
    checkCuda(cudaMemcpyToSymbol(clsl::lights, &backend->lights_buffer, sizeof(clsl::lights)));
    checkCuda(cudaMemcpyToSymbol(clsl::textures, &backend->sampler_buffer, sizeof(clsl::textures)));

    checkCuda(cudaMemcpyToSymbol(clsl::view_params, &accumulated_spp, sizeof(accumulated_spp), offsetof(clsl::ViewParams, frame_id)));

    checkCuda(cudaDeviceSynchronize());
}
#endif
