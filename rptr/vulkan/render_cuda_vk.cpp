#include "render_cuda_vk.h"
#include <cstring>
#include <cassert>
#include <unordered_map>
#include "scene.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

namespace vkrt {
#ifdef _WIN32
    extern PFN_vkGetMemoryWin32HandleKHR GetMemoryWin32HandleKHR;
#else
    extern PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
#endif
}

namespace {

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice, cudaDeviceProp* deviceProp) {
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice, &vkPhysicalDeviceProperties2);

    int cudaDeviceCount = 0;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaGetDeviceProperties(deviceProp, cudaDevice);
        if (!memcmp(deviceProp->uuid.bytes, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);
    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
    return extMem;
}

#ifdef _WIN32
cudaExternalMemory_t importVulkanMemoryObjectFromWindowsHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}
#endif

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    // Note: ‘ptr’ must eventually be freed using cudaFree()
    return ptr;
}

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc const& formatDesc, cudaExtent const& extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    desc.offset = offset;
    desc.formatDesc = formatDesc;
    desc.extent = extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: ‘mipmap’ must eventually be freed using cudaFreeMipmappedArray()
    checkCuda( cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc) );
    assert(mipmap);
    return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format) {
    cudaChannelFormatDesc d = {};

    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8_UINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8_SINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8_UINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8_SINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8B8A8_SINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16_UINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16_SINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16G16_UINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16G16_SINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16G16B16A16_UINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16G16B16A16_SINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32_UINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32_SINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32_SFLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case VK_FORMAT_R32G32_UINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32G32_SINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32G32_SFLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case VK_FORMAT_R32G32B32A32_UINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32G32B32A32_SINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;

    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:  d.x = 32;  d.y = 32;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:     d.x = 32;  d.y = 32;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:  d.x = 32;  d.y = 32;  d.z = 32;  d.w = 32;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:  d.x = 32;  d.y = 32;  d.z = 32;  d.w = 32;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK: d.x = 32;  d.y = 32;  d.z = 32;  d.w = 32;  d.f = cudaChannelFormatKindUnsigned; break;
    default: assert(0);
    }

    return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType, VkFormat format) {
    cudaExtent e = { 0, 0, 0 };
 
    switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_1D:         e.width = vkExt.width; e.height = 0;            e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_2D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_3D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:       e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   e.width = vkExt.width; e.height = 0;            e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    default: assert(0);
    }

    switch (format) {
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
        e.width /= 4;
        e.height /= 4;
        break;
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
        e.width /= 4;
        e.height /= 4;
        break;
    default: break;
    }

    return e;
}

unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;

    switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_CUBE:       flags |= cudaArrayCubemap;                    break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   flags |= cudaArrayLayered;                    break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   flags |= cudaArrayLayered;                    break;
    default: break;
    }

    if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        flags |= cudaArrayColorAttachment;
    }

    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }
    return flags;
}

cudaExternalMemory_t import_memory(VkDevice logical_device, VkDeviceMemory memory, size_t size) {
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle = nullptr;
    info.memory = memory;
    CHECK_VULKAN( vkrt::GetMemoryWin32HandleKHR(logical_device, &info, &handle) );
 
    cudaExternalMemory_t mem = importVulkanMemoryObjectFromWindowsHandle(handle,size,false);
#else
    VkMemoryGetFdInfoKHR fdq = { };
    fdq.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdq.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = 0;
    fdq.memory = memory;
    CHECK_VULKAN( vkrt::GetMemoryFdKHR(logical_device, &fdq, &fd) );

    cudaExternalMemory_t mem = importVulkanMemoryObjectFromFileDescriptor(fd, size, false);
#endif
    return mem;
}

struct VulkanMemoryImporter {
    vkrt::Device& device;
    std::vector<cudaExternalMemory_t>& imported_memory_objects;

    struct ImportedMemory {
        cudaExternalMemory_t memory = NULL;
        size_t size = 0;
    };
    std::unordered_map<VkDeviceMemory, ImportedMemory> external_memory;

    VulkanMemoryImporter(vkrt::Device& device
        , std::vector<cudaExternalMemory_t>& imported_memory_objects)
        : device(device)
        , imported_memory_objects(imported_memory_objects) {
    }
    void register_arena(uint32_t arena_idx) {
        auto blocks = device.blocks_in_arena(arena_idx, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        for (auto block : blocks)
            external_memory[block.memory].size = block.size;
    }
    cudaExternalMemory_t import(VkDeviceMemory memory, size_t size) {
        auto it = external_memory.find(memory);
        if (it != external_memory.end()) {
            if (it->second.memory)
                return it->second.memory;
            else
                size = it->second.size;
        }
        cudaExternalMemory_t cuda_memory = import_memory(device.logical_device(), memory, size);
        assert(cuda_memory);
        if (it != external_memory.end())
            it->second.memory = cuda_memory;
        else
            external_memory[memory] = { cuda_memory, size };
        imported_memory_objects.push_back(cuda_memory);
        return cuda_memory;
    };
};

} // namespace

RenderCudaBinding* create_vulkan_cuda_extension(RenderBackend* vulkan_backend) {
    return new RenderVulkanCuda(&dynamic_cast<RenderVulkan&>(*vulkan_backend));
}

RenderVulkanCuda::RenderVulkanCuda(RenderVulkan* backend)
    : RenderCudaBinding(backend) {
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA runtime version: %d\n", runtimeVersion);

    alignas(64) cudaDeviceProp props; // todo: why crazy alignment required in DEBUG?
    this->cudaDeviceId = getCudaDeviceForVulkanPhysicalDevice(backend->device.physical_device(), &props);

    cudaSetDevice(this->cudaDeviceId);

    if (props.concurrentKernels)
        num_concurrent_streams = MAX_CONCURRENT_KERNELS;
    else
        num_concurrent_streams = 1;
    printf("Using %d concurrent CUDA streams\n", num_concurrent_streams);

    for (int i = 0; i < num_concurrent_streams; ++i)
        cudaStreamCreate(&concurrent_streams[i]);
}

RenderVulkanCuda::~RenderVulkanCuda() {
    release_mapped_resources();

    for (int i = 0; i < num_concurrent_streams; ++i)
        checkCuda(cudaStreamDestroy(concurrent_streams[i]));
}

void RenderVulkanCuda::release_mapped_resources() {
    release_mapped_scene_resources(nullptr);
    release_mapped_display_resources();

    for (auto mem : persistent_external_memory)
        checkCuda(cudaDestroyExternalMemory(mem));
    persistent_external_memory.clear();
}

void RenderVulkanCuda::release_mapped_scene_resources(const Scene* release_changes_only) {
    // for (auto& mesh : this->meshes) {
        // for (auto& geom : mesh.geometries) {
        //     checkCuda(cudaFree(geom.vertices));
        //     geom.vertices = nullptr;
        //     checkCuda(cudaFree(geom.indices));
        //     geom.indices = nullptr;
        //     checkCuda(cudaFree(geom.normals));
        //     geom.normals = nullptr;
        //     checkCuda(cudaFree(geom.uvs));
        //     geom.uvs = nullptr;
        // }
    // }
    for (auto& pointer : scene_external_pointers) {
        checkCuda(cudaFree(pointer));
    }
    scene_external_pointers.clear();

    for (auto& pmesh : this->parameterized_meshes) {
        checkCuda(cudaFree(pmesh.perTriangleMaterialIDs));
        pmesh.perTriangleMaterialIDs = nullptr;
    }
    this->meshes.clear();
    this->parameterized_meshes.clear();

    release_sampler_objects();
    for (auto& tex : this->textures) {
        checkCuda(cudaFreeMipmappedArray(tex));
        tex = nullptr;
    }
    this->textures.clear();

    checkCuda(cudaFree(this->global_params_buffer));
    this->global_params_buffer = nullptr;
    checkCuda(cudaFree(this->instanced_geometry_buffer));
    this->instanced_geometry_buffer = nullptr;
    checkCuda(cudaFree(this->material_buffer));
    this->material_buffer = nullptr;
    checkCuda(cudaFree(this->lights_buffer));
    this->lights_buffer = nullptr;

    for (auto mem : scene_external_memory)
        checkCuda(cudaDestroyExternalMemory(mem));
    scene_external_memory.clear();
}

void RenderVulkanCuda::release_mapped_display_resources() {
    for (int i = 0; i < 2; ++i)
        cudaDestroySurfaceObject(this->pixel_buffers[i]);
    for (int i = 0; i < 2; ++i)
        cudaDestroySurfaceObject(this->accum_buffers[i]);
    cudaDestroySurfaceObject(this->debug_buffer);

    for (auto& tex : display_textures)
        checkCuda(cudaFreeMipmappedArray(tex));
    display_textures.clear();

    checkCuda(cudaFree(this->ray_queries));
    this->ray_queries = nullptr;
    checkCuda(cudaFree(this->ray_results));
    this->ray_results = nullptr;

    for (auto mem : display_external_memory)
        checkCuda(cudaDestroyExternalMemory(mem));
    display_external_memory.clear();
}

std::string RenderVulkanCuda::name() const {
    return "Vulkan Cuda Extension";
}
    
void RenderVulkanCuda::initialize(const int fb_width, const int fb_height) {
    release_mapped_display_resources();

    RenderCudaBinding::initialize(fb_width, fb_height);

    RenderVulkan *backend = this->vulkan_backend();

    VulkanMemoryImporter memory_importer(backend->device, display_external_memory);
    memory_importer.register_arena(vkrt::Device::DisplayArena);

    auto&& import_texture = [&memory_importer](vkrt::Texture2D tex) -> cudaMipmappedArray_t {
        if (!tex) return nullptr;
        VkExtent3D ext = { uint_bound(tex->tdims.x), uint_bound(tex->tdims.y), 0 };
        cudaExternalMemory_t mem = memory_importer.import(tex.ref_data->mem, tex.ref_data->mem_size);
        cudaMipmappedArray_t cutex = mapMipmappedArrayOntoExternalMemory(mem, tex.ref_data->mem_offset
            , getCudaChannelFormatDescForVulkanFormat(tex->pixel_format())
            , getCudaExtentForVulkanExtent(ext, 1, VK_IMAGE_VIEW_TYPE_2D, tex->pixel_format())
            , getCudaMipmappedArrayFlagsForVulkanImage(VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true)
            , 1);
        assert(cutex);
        return cutex;
    };

    for (int i = 0; i < 2; ++i) {
        display_textures.push_back({});
        auto& accum_tex = display_textures.back();
        accum_tex = import_texture(backend->accum_buffers[i]);
        cudaResourceDesc resDesc = { };
        resDesc.resType = cudaResourceTypeArray;
        cudaGetMipmappedArrayLevel(&resDesc.res.array.array, accum_tex, 0);
        assert(resDesc.res.array.array);
        cudaCreateSurfaceObject(&this->accum_buffers[i], &resDesc);
        assert(this->accum_buffers[i]);
    }

    for (int i = 0; i < 2; ++i) {
        display_textures.push_back({});
        auto& pixel_tex = display_textures.back();
        pixel_tex = import_texture(backend->render_targets[i]);
        cudaResourceDesc resDesc = { };
        resDesc.resType = cudaResourceTypeArray;
        cudaGetMipmappedArrayLevel(&resDesc.res.array.array, pixel_tex, 0);
        assert(resDesc.res.array.array);
        cudaCreateSurfaceObject(&this->pixel_buffers[i], &resDesc);
        assert(this->pixel_buffers[i]);
    }

    debug_textures.push_back({});
    auto& debug_tex = debug_textures.back();
    debug_tex = import_texture(backend->debug_texture_buffer);
    cudaResourceDesc resDesc = { };
    cudaGetMipmappedArrayLevel(&resDesc.res.array.array, debug_tex, 0);
    assert(resDesc.res.array.array);
    cudaCreateSurfaceObject(&this->debug_buffer, &resDesc);
    assert(this->debug_buffer);

    auto&& import_buffer = [&memory_importer](vkrt::Buffer buf) -> void* {
        if (!buf) return nullptr;
        cudaExternalMemory_t mem = memory_importer.import(buf.ref_data->mem, buf.ref_data->buf_size);
        void* device_ptr = mapBufferOntoExternalMemory(mem, buf.ref_data->mem_offset, buf->size());
        assert(device_ptr);
        return device_ptr;
    };
    ray_queries = (RenderRayQuery*) import_buffer(backend->ray_query_buffer);
    ray_results = (RenderRayQueryResult*) import_buffer(backend->ray_result_buffer);
}

void RenderVulkanCuda::update_scene_from_backend(const Scene& scene) {
    RenderCudaBinding::update_scene_from_backend(scene);

    RenderVulkan *backend = this->vulkan_backend();

    VulkanMemoryImporter memory_importer(backend->device, scene_external_memory);
    memory_importer.register_arena(vkrt::Device::PersistentArena);
    memory_importer.register_arena(backend->base_arena_idx + RenderVulkan::StaticArenaOffset);
    memory_importer.register_arena(backend->base_arena_idx + RenderVulkan::DynamicArenaOffset);

    auto&& import_buffer = [&memory_importer,this](vkrt::Buffer buf, bool scene_params = false) -> void* {
        if (!buf) return nullptr;
        cudaExternalMemory_t mem = memory_importer.import(buf.ref_data->mem, buf.ref_data->buf_size);
        void* device_ptr = mapBufferOntoExternalMemory(mem, buf.ref_data->mem_offset, buf->size());
        assert(device_ptr);
        if (scene_params)
            this->scene_external_pointers.push_back(device_ptr);
        return device_ptr;
    };
    auto&& import_texture = [&memory_importer](vkrt::Texture2D tex) -> cudaMipmappedArray_t {
        if (!tex) return nullptr;
        VkExtent3D ext = { uint_bound(tex->tdims.x), uint_bound(tex->tdims.y), 1 };
        cudaExternalMemory_t mem = memory_importer.import(tex.ref_data->mem, tex.ref_data->mem_size);
        cudaMipmappedArray_t cutex = mapMipmappedArrayOntoExternalMemory(mem, tex.ref_data->mem_offset
            , getCudaChannelFormatDescForVulkanFormat(tex->pixel_format())
            , getCudaExtentForVulkanExtent(ext, 1, VK_IMAGE_VIEW_TYPE_2D, tex->pixel_format())
            , getCudaMipmappedArrayFlagsForVulkanImage(VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT, false)
            , tex->mips);
        assert(cutex);
        return cutex;
    };

    for (auto const& mesh : backend->meshes) {
        this->meshes.push_back({});
        auto& cumesh = this->meshes.back();
        for (auto const& geom : mesh->geometries) {

#ifdef QUANTIZED_POSITIONS
            int vertex_stride = sizeof(uint64_t);
#else
            int vertex_stride = sizeof(float) * 3;
#endif
#ifdef QUANTIZED_NORMALS_AND_UVS
            int normal_stride = sizeof(uint64_t);
            int uv_stride = sizeof(uint64_t);
#else
            int normal_stride = sizeof(float) * 3;
            int uv_stride = sizeof(float) * 2;
#endif

            int vertex_offset = geom.vertex_offset;

            cumesh.geometries.push_back({});
            auto& cugeom = cumesh.geometries.back();

            cugeom.vertexCount = geom.num_vertices();
            cugeom.triangleCount = geom.num_triangles();

            if (!geom.indices_are_implicit){
                cugeom.indices = (glm::uvec3*)import_buffer(geom.index_buf, true) + geom.triangle_offset;
                vertex_offset += geom.index_offset;
            }

            cugeom.vertices = (char*)import_buffer(geom.vertex_buf, true) + vertex_offset*vertex_stride;
            cugeom.normals = (char*)import_buffer(geom.normal_buf, true) + vertex_offset*normal_stride;
            cugeom.uvs = (char*)import_buffer(geom.uv_buf,true) + vertex_offset*uv_stride;

        }
    }

    for (auto const& pmesh : backend->parameterized_meshes) {
        this->parameterized_meshes.push_back({});
        auto& cumesh = this->parameterized_meshes.back();
        cumesh.perTriangleMaterialIDs = import_buffer(pmesh.per_triangle_material_buf);
        cumesh.mesh_id = pmesh.mesh_id;
        cumesh.no_alpha = pmesh.no_alpha;
    }

    this->local_host_params = backend->local_params();
    this->global_params_buffer = import_buffer(backend->global_param_buf);
    this->material_buffer = import_buffer(backend->mat_params);
    this->lights_buffer = import_buffer(backend->binned_light_params);

    for (auto const& tex : backend->textures) {
        this->textures.push_back({});
        this->textures.back() = import_texture(tex);
    }
    create_sampler_objects(scene);

    std::vector<InstancedGeometry> instances;
    {
        std::vector< std::vector<RenderMeshParams> > geometries = backend->render_meshes;
        for (int pm_idx = 0; pm_idx < (int) scene.parameterized_meshes.size(); ++pm_idx) {
            const auto& pm = scene.parameterized_meshes[pm_idx];
            const auto& mesh = scene.meshes[pm.mesh_id];
            auto& geo_params = geometries[pm_idx];
            len_t primOffset = 0;
            for (int geo_idx = 0; geo_idx < (int) mesh.geometries.size(); ++geo_idx) {
                //const auto &geo = mesh.geometries[geo_idx];
                const auto &cugeo = this->meshes[pm.mesh_id].geometries[geo_idx];
                RenderMeshParams &rmp = geo_params[geo_idx];
                rmp.indices.i = (decltype(rmp.indices.i)) cugeo.indices;
                rmp.vertices.v = (decltype(rmp.vertices.v)) cugeo.vertices;
                rmp.normals.n = (decltype(rmp.normals.n)) cugeo.normals;
                rmp.uvs.uv = (decltype(rmp.uvs.uv)) cugeo.uvs;
                rmp.materials.id_4pack = (decltype(rmp.materials.id_4pack)) ((char*) this->parameterized_meshes[pm_idx].perTriangleMaterialIDs + primOffset);
                primOffset += cugeo.triangleCount;
            }
        }

        for (const auto& inst : backend->instances) {
            auto const& geoms = geometries[inst.parameterized_mesh_id];

            int inst_geom_idx = ilen(instances);
            instances.resize(instances.size() + geoms.size());

            glm::mat4 m = inst.transform;
            glm::mat4 mI = inverse(m);

            for (auto const& g : geoms) {
                auto& i = instances[inst_geom_idx++];
                i.instance_to_world = m;
                i.world_to_instance = mI;
                i.geometry = g;
            }
        }
    }
    checkCuda(cudaMalloc((void**) &this->instanced_geometry_buffer, sizeof(InstancedGeometry) * instances.size()));
    checkCuda(cudaMemcpy(this->instanced_geometry_buffer, instances.data(), sizeof(InstancedGeometry) * instances.size(), cudaMemcpyHostToDevice));
}

int RenderVulkanCuda::active_pixel_buffer_index() {
    return static_cast<RenderVulkan*>(backend)->active_render_target;
}

int RenderVulkanCuda::active_accum_buffer_index() {
    return static_cast<RenderVulkan*>(backend)->active_accum_buffer;
}
