#define INCLUDE_BASE_WAVEFRONT_PT
#include "wavefront_pt.cuh" // Wavefront PT device environment

namespace clsl {

__global__ void camera_raygen(int sample_idx, int rnd_sample_offset
    , RenderRayQuery *queries, glm::vec4 *throughtput_pdf, glm::vec4 *illumination_rnd) {
    uvec2 pixel = WavefrontPixelID;
    //uvec2 pixel = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int i = int(pixel.y) * fb_width + int(pixel.x);
    if (pixel.x >= unsigned(fb_width) || pixel.y >= unsigned(fb_height)) return;

    auto lcg = get_lcg_rng(sample_idx, rnd_sample_offset, i);
    float xf = (float(pixel.x) + lcg_randomf(lcg)) / float(fb_width);
    float yf = (float(pixel.y) + lcg_randomf(lcg)) / float(fb_height);

    camera_raygen(i, xf, yf, lcg.state,
        queries, throughtput_pdf, illumination_rnd);
}


__global__ void bounce_ray(int bounce, WavefrontBounceData data) {
    int i = WavefrontPixelIndex;
    if (i >= fb_width * fb_height) return;

#ifdef CAST_SHADOW_RAYS
    // note: this seems terribly unnecessary, find a better way to do this w/o going to explicit SIMD?
    data.shadow_queries[i].mode_or_data = -1;
#endif

    bool path_active = data.bounce_queries[i].mode_or_data >= 0;
    if (path_active) {
        path_active = bounce_ray(i, bounce, data);
        if (!path_active)
            data.bounce_queries[i].mode_or_data = -1;
    }
}

__global__ void postprocess_bounce(int bounce, WavefrontBounceData data) {
    int i = WavefrontPixelIndex;
    if (i >= fb_width * fb_height) return;

    bool shadow_active = data.shadow_queries[i].mode_or_data >= 0;
    if (shadow_active) {
        postprocess_bounce(i, bounce, data.illumination_rnd, data.nee_pdf, data.shadow_results);
    }
}

__global__ void accum_results(int sample_index, glm::vec4 *illumination_rnd) {
    //uvec2 pixel = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    uvec2 pixel = WavefrontPixelID;
    int i = int(pixel.y) * fb_width + int(pixel.x);
    if (pixel.x >= unsigned(fb_width) || pixel.y >= unsigned(fb_height)) return;

    vec3 illum = vec3(illumination_rnd[i]);

    vec3 accum_color = vec3(0.0f);
    if (sample_index > 0) {
        float4 data;
        surf2Dread(&data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
        accum_color = vec3(data.x, data.y, data.z);
    }
    accum_color += illum;
    {
        float4 data = { accum_color.x, accum_color.y, accum_color.z, render_params.output_channel == 0 ? 1.0f : -1.0f };
        surf2Dwrite(data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
    }

    accum_color /= float(sample_index + 1);

    if (render_params.output_channel == 0) {
        accum_color *= exp2(render_params.exposure);
        float luminance_level = max(max(accum_color.x, accum_color.y), max(accum_color.z, 1.0f));
        accum_color *= mix(0.1f * log2(luminance_level), 1.0f, 0.8f) / luminance_level;
    }
    else if (render_params.output_channel == 2) {
        if (render_params.output_moment != 0)
            accum_color = vec3(length(accum_color));
        else
            accum_color = accum_color * 0.5f + vec3(0.5f);
    }
    else if (render_params.output_channel == 3) {
        accum_color = (accum_color - vec3(view_params.cam_pos)) * 0.1f + vec3(0.5f);
    }

    accum_color = vec3(linear_to_srgb(accum_color.x), linear_to_srgb(accum_color.y), linear_to_srgb(accum_color.z));
    {
        uchar4 data;
        data.x = glm::clamp( int( accum_color.x * 256.0f ), 0, 255 );
        data.y = glm::clamp( int( accum_color.y * 256.0f ), 0, 255 );
        data.z = glm::clamp( int( accum_color.z * 256.0f ), 0, 255 );
        data.w = 255;
        surf2Dwrite(data, framebuffer, pixel.x * 4, pixel.y, cudaBoundaryModeZero);
    }
}

} // namespace

clsl::WavefrontBounceData WavefrontPT::bounce_data(int bounce) {
    int screen_query_count = backend->screen_width * backend->screen_height;

    clsl::WavefrontBounceData data = { };
    data.bounce_queries = backend->ray_queries;
    data.bounce_results = backend->ray_results;
    data.throughtput_pdf = this->throughtput_pdf;
    data.illumination_rnd = this->illumination_rnd;
    data.shadow_queries = backend->ray_queries + screen_query_count;
    data.shadow_results = backend->ray_results + screen_query_count;
    data.nee_pdf = this->nee_pdf;
    return data;
}

void WavefrontPT::setup_constants() {
    WavefrontPT::this_module_setup_constants();
}

void WavefrontPT::camera_raygen() {
    dim3 dimBlock(32, 16);
    dim3 dimGrid((backend->screen_width  + dimBlock.x - 1) / dimBlock.x,
                 (backend->screen_height + dimBlock.y - 1) / dimBlock.y);
    clsl::camera_raygen<<<dimGrid, dimBlock>>>(accumulated_spp, sample_offset
        , backend->ray_queries, this->throughtput_pdf, this->illumination_rnd);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void WavefrontPT::cast_rays(int bounce, bool shadow_rays) {
    int screen_query_count = backend->screen_width * backend->screen_height;
    int total_query_count = screen_query_count;
    if (shadow_rays)
        total_query_count += screen_query_count;

    backend->backend->render_ray_queries(total_query_count
        , backend->backend->params, rt_rayquery_index);
}

void WavefrontPT::bounce_rays(int bounce) {
    clsl::WavefrontBounceData data = bounce_data(bounce);

    dim3 dimBlock(32, 16);
    dim3 dimGrid((backend->screen_width  + dimBlock.x - 1) / dimBlock.x,
                 (backend->screen_height + dimBlock.y - 1) / dimBlock.y);
    clsl::bounce_ray<<<dimGrid, dimBlock>>>(bounce, data);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void WavefrontPT::postprocess_bounce(int bounce) {
    clsl::WavefrontBounceData data = bounce_data(bounce);

    dim3 dimBlock(32, 16);
    dim3 dimGrid((backend->screen_width  + dimBlock.x - 1) / dimBlock.x,
                 (backend->screen_height + dimBlock.y - 1) / dimBlock.y);
    clsl::postprocess_bounce<<<dimGrid, dimBlock>>>(bounce, data);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void WavefrontPT::accumulate_results(int sample_idx) {
    dim3 dimBlock(32, 16);
    dim3 dimGrid((backend->screen_width  + dimBlock.x - 1) / dimBlock.x,
                 (backend->screen_height + dimBlock.y - 1) / dimBlock.y);
    clsl::accum_results<<<dimGrid, dimBlock>>>(sample_idx, this->illumination_rnd);
    ++accumulated_spp;

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void WavefrontPT::render(bool reset_accumulation) {
    setup_constants();

    if (reset_accumulation) {
        sample_offset += accumulated_spp;
        accumulated_spp = 0;
    }

    for (int spp = 0; spp < backend->backend->params.batch_spp; ++spp) {

    camera_raygen();
    for (int bounce = 0; bounce < backend->backend->params.max_path_depth; ++bounce) {
        bool shadow_rays = false;

        int prev_bounce = bounce - 1;
        if (prev_bounce >= 0) {
#ifdef CAST_SHADOW_RAYS
            shadow_rays = backend->backend->params.output_channel == 0;
#endif
        }

        cast_rays(bounce, shadow_rays);

        if (prev_bounce >= 0)
            postprocess_bounce(prev_bounce);

        bounce_rays(bounce);
    }
    accumulate_results(spp);

    }
}
