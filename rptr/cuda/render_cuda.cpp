#include "render_cuda.h"
#include "scene.h"

RenderCudaBinding::RenderCudaBinding(RenderBackend* backend)
    : backend(backend) {
}

RenderCudaBinding::~RenderCudaBinding() {
    release_sampler_objects();
}

std::string RenderCudaBinding::name() const {
    return "Cuda Extension";
}

void RenderCudaBinding::initialize(const int fb_width, const int fb_height) {
    this->screen_width = fb_width;
    this->screen_height = fb_height;
}

void RenderCudaBinding::update_scene_from_backend(const Scene& scene) {
}

void RenderCudaBinding::release_sampler_objects() {
    if (sampler_buffer) {
        checkCuda(cudaFree(sampler_buffer));
        sampler_buffer = nullptr;
    }

    for (auto tex : samplers) {
        if (tex)
            checkCuda(cudaDestroyTextureObject(tex));
    }
    samplers.clear();
}

void RenderCudaBinding::create_sampler_objects(const Scene& scene) {
    release_sampler_objects();

    assert(textures.size() == scene.textures.size());
    samplers.resize(textures.size(), 0);
    for (int i = 0, ie = ilen(textures); i < ie; ++i) {
        ::Image const& image = scene.textures[i];
        cudaMipmappedArray_t array = textures[i];
        if (!array)
            continue;
        
        cudaTextureObject_t sampler = 0;

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = array;

        cudaResourceViewDesc res_view_desc = { };
        res_view_desc.width = image.width;
        res_view_desc.height = image.height;
        res_view_desc.lastMipmapLevel = image.mip_levels() - 1;
        switch (image.bcFormat) {
        case -1:
        case 1: res_view_desc.format = cudaResViewFormatUnsignedBlockCompressed1; break;
        case -2:
        case 2: res_view_desc.format = cudaResViewFormatUnsignedBlockCompressed2; break;
        case -3:
        case 3: res_view_desc.format = cudaResViewFormatUnsignedBlockCompressed3; break;
        case -4: res_view_desc.format = cudaResViewFormatSignedBlockCompressed4; break;
        case 4: res_view_desc.format = cudaResViewFormatUnsignedBlockCompressed4; break;
        case -5: res_view_desc.format = cudaResViewFormatSignedBlockCompressed5; break;
        case 5: res_view_desc.format = cudaResViewFormatUnsignedBlockCompressed5; break;
        default: res_view_desc.format = cudaResViewFormatUnsignedChar4; break;
        }

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = image.bcFormat ? cudaFilterModePoint : cudaFilterModeLinear; // can only point sample due to CUDA bug with block compression validation?
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.readMode = image.bcFormat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
        tex_desc.sRGB = image.color_space == SRGB ? 1 : 0;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.maxMipmapLevelClamp = 0x7fff;
        checkCuda(cudaCreateTextureObject(&sampler, &res_desc, &tex_desc, &res_view_desc));
        assert(sampler);

        samplers[i] = sampler;
    }

    checkCuda(cudaMalloc(&sampler_buffer, sizeof(*sampler_buffer) * samplers.size()));
    checkCuda(cudaMemcpy(sampler_buffer, samplers.data(), sizeof(*sampler_buffer) * samplers.size(), cudaMemcpyHostToDevice));
}
