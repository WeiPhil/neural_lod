/* 
 * Copyright 2022 Intel Corporation.
 */

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Functions return these codes to indicate various errors.
 */
typedef enum {
  VKR_SUCCESS             = 0,
  VKR_INVALID_ARGUMENT    = -1,
  VKR_INVALID_FILE_NAME   = -2,
  VKR_INVALID_FILE_FORMAT = -3,
  VKR_ALLOCATION_ERROR    = -4,
  VKR_MMAP_ERROR          = -5,
  VKR_INVALID_TEXTURE_FORMAT = -6,
  VKR_RESULT_MAX_ENUM     = 0x7FFFFFFF
} VkrResult;

/*
 * Optionally, pass in a pointer to an error handling function to
 * receive detailed error messages.
 */
typedef void (*VkrErrorHandler)(VkrResult result, const char *msg);


/*
 * Material ID size -- enum value is size in bytes.
 */
typedef enum {
  VKR_MATERIAL_ID_8_BITS  = 1,
  VKR_MATERIAL_ID_16_BITS = 2 // will be deprecated
} VkrMaterialIdSize;


/*
 * Enum compatible with Vulkan's texture format enum,
 * VK_FORMAT_BC*...
 */
typedef enum {
  // 8 byte per 4x4 block with linear RGB data
  VKR_TEXTURE_FORMAT_BC1_RGB_UNORM_BLOCK = 131,
  // 8 byte per 4x4 block with sRGB data
  VKR_TEXTURE_FORMAT_BC1_RGB_SRGB_BLOCK = 132,
  // 8 byte per 4x4 block with linear RGBA data, 1 bit alpha
  VKR_TEXTURE_FORMAT_BC1_RGBA_UNORM_BLOCK = 133,
  // 8 byte per 4x4 block with sRGBA data, 1 bit alpha
  VKR_TEXTURE_FORMAT_BC1_RGBA_SRGB_BLOCK = 134,
  // 16 byte per 4x4 block with linear RGBA data, 8 bit alpha
  VKR_TEXTURE_FORMAT_BC3_UNORM_BLOCK = 137,
  // 16 byte per 4x4 block with sRGBA data, 8 bit alpha
  VKR_TEXTURE_FORMAT_BC3_SRGB_BLOCK = 138,
  // 16 byte per 4x4 block with two channel linear data
  VKR_TEXTURE_FORMAT_BC5_UNORM_BLOCK = 141,
  // 4 byte per texel with linear data
  VKR_TEXTURE_FORMAT_R8G8B8A8_UNORM = 37,
} VkrTextureFormat;


/*
 * Descriptor for a single mip level.
 */
typedef struct {
  int32_t width;
  int32_t height;
  uint64_t dataSize; // In bytes.
  int64_t dataOffset; // In bytes, in the file.
} VkrMipLevel;


/*
 * A texture loaded from a .vkrt file.
 */
typedef struct {
  const char *filename;
  int32_t version;
  int32_t width;
  int32_t height;
  int32_t format;  // Compatible with Vulkan's VkFormat!
  int32_t numMipLevels;
  const VkrMipLevel *mipLevels;
  uint64_t dataSize; // ...of the full mip data, in bytes.
  int64_t dataOffset; // ...of the full mip data, in bytes, in the file.
} VkrTexture;


/*
 * Tensor format.
 */
typedef enum {
  VKR_TENSOR_FORMAT_HALF_FLOAT  = 1,
  VKR_TENSOR_FORMAT_FLOAT = 2
} VkrTensorFormat;

/*
 * Tensor flags.
 */
typedef enum {
  VKR_TENSOR_FLAGS_NONE = 0,
  // tensor includes input and output blocks besides fixed-size inner layers.
  VKR_TENSOR_FLAGS_INPUT_OUTPUT_SPEC = 0x1,
  // output tensor is stored in a transposed format that ensures compactness
  // of relevant weights and compatible layout with varying output sizes.
  VKR_TENSOR_FLAGS_OUTPUT_TRANSPOSED = 0x2,
  // tensor comes with implicit biases added as part of the input multiplication
  VKR_TENSOR_FLAGS_IMPLICIT_BIASES = 0x4,
  // the array of elements may not match the standard layout of full tensors
  // as described by the header (e.g. sparse layouts). Correct interpretation
  // is left up to the application.
  VKR_TENSOR_FLAGS_CUSTOM_DATA_LAYOUT = 0x8,
  // mask that can be used to check if this tensor describes a neural network.
  VKR_TENSOR_FLAGS_NEURAL_MASK = VKR_TENSOR_FLAGS_INPUT_OUTPUT_SPEC | VKR_TENSOR_FLAGS_OUTPUT_TRANSPOSED,
  VKR_TENSOR_FLAGS_MAX_ENUM = 0x7FFFFFFF
} VkrTensorFlags;

/*
 * A tensor definition.
 */
typedef struct {
  #define VkrTensorMaxDimensionality 4
  uint64_t dimensionality;
  VkrTensorFormat format;
  uint32_t flags;
  uint64_t dimensions[VkrTensorMaxDimensionality];
  uint64_t numInputs;
  uint64_t numInputLayerBlocks;
  uint64_t numOutputs;
  uint64_t numOutputLayerBlocks;
  uint64_t storageDescriptor;
  uint64_t componentsDescriptor;
  double   ratioDescriptor;
  uint64_t numValues;
  void const *values;
  uint64_t dataSize;
} VkrTensor;

/*
 * A material definition.
 * NOTE: If you add members here, make sure to also load the corresponding
 *       .txt files in vkr_load_material().
 */
typedef struct {
  const char *name;

  float baseColor[3];
  VkrTexture texBaseColor;

  float normal[3];
  VkrTexture texNormal;

  float ao;
  float metalness;
  float roughness;
  VkrTexture texAoMetalnessRoughness;

  float emissionIntensity;
  float specularTransmission;
  float iorEta;
  float iorK;
  float translucency;

  #define VkrMaterialMaxFeatureTextures 4
  VkrTexture features[VkrMaterialMaxFeatureTextures];

  #define VkrMaterialMaxTensors 3
  VkrTensor tensors[VkrMaterialMaxTensors];
} VkrMaterial;

/*
 * Mesh flags.
 */
typedef enum {
  VKR_MESH_FLAGS_NONE        = 0,
  VKR_MESH_FLAGS_INDICES     = 0x1,
  VKR_MESH_FLAGS_MAX_ENUM    = 0x7FFFFFFF
} VkrMeshFlags;

/*
 * A mesh definition.
 */
typedef struct {
  const char *name;
  float vertexScale[3];
  float vertexOffset[3];
  float scaleBoundsMin[3]; // AABB minimum inferred from vertex offset.
  float scaleBoundsMax[3]; // AABB maximum inferred from vertex offset and scale.

  uint32_t flags;
  uint64_t numSegments;
  int32_t materialIdBufferBase;
  uint32_t numMaterialsInRange;
  uint64_t numTriangles;

  int64_t vertexBufferOffset; // In bytes, in the file.
  int64_t normalUvBufferOffset; // In bytes, in the file.

  // There are numTriangles material IDs.
  int64_t materialIdBufferOffset; // In bytes, in the file.
  VkrMaterialIdSize materialIdSize; // In bytes (one id is this big).

  // Optionally, there are numTriangles 32-bit vertex sharing indices.
  int64_t indexBufferOffset; // In bytes, in the file.

  uint64_t *segmentNumTriangles;
  int32_t *segmentMaterialBaseOffsets;
} VkrMesh;


/*
 * An instance definition.
 */
typedef struct {
  int64_t headerSize; // In bytes, in the file (primarily for internal use).

  const char *name;
  int64_t meshId;
  float transform[4][3];

  uint32_t flags;
} VkrInstance;


/*
 * Initialize one of these structs with vkr_open, and
 * make sure to call vkr_close for cleanup.
 */
typedef struct {
  int32_t version;
  uint32_t flags; // Reserved
  int64_t headerSize;
  int64_t dataOffset;

  const char *textureDir;

  uint64_t numMaterials;
  VkrMaterial *materials;

  uint64_t numTriangles;
  uint64_t numMeshes;
  VkrMesh *meshes;

  uint64_t numInstances;
  VkrInstance *instances;
} VkrScene;


/*
 * Open the texture file pointed to by filename.
 *
 * Will return VKR_SUCCESS on success, and fill the VkrTexture struct.
 *
 * The error handler is optional, you may pass NULL instead.
 * Note: vkr_open_texture will return VKR_INVALID_FILE_NAME if it fails
 *       to open the texture file. However, the error handler will not be
 *       called in this case.
 *       This is because textures are generally optional.
 *
 * On failure, t->filename will be NULL.
 */
VkrResult vkr_open_texture(
    const char *filename,
    VkrTexture *t,
    VkrErrorHandler errorHandler);

/*
 * Close the texture.
 */
void vkr_close_texture(VkrTexture *t);


/*
 * Open the tensor file pointed to by filename.
 *
 * Will return VKR_SUCCESS on success, and fill the VkrTensor struct.
 *
 * The error handler is optional, you may pass NULL instead.
 * Note: vkr_open_tensor will return VKR_INVALID_FILE_NAME if it fails
 *       to open the tensor file. However, the error handler will not be
 *       called in this case.
 *       This is because tensors are generally optional.
 */
VkrResult vkr_open_tensor(
    const char *filename,
    VkrTensor *t,
    VkrErrorHandler errorHandler);

/*
 * Close the tensor.
 */
void vkr_close_tensor(VkrTensor *t);

\
/*
 * Open the scene file pointed to by filename.
 *
 * Will return VKR_SUCCESS on success, and fill the VkrScene struct.
 *
 * The error handler is optional, you may pass NULL instead.
 */
VkrResult vkr_open_scene(const char *filename, VkrScene *v,
    VkrErrorHandler errorHandler);

/*
 * Close the scene.
 */
void vkr_close_scene(VkrScene *v);

/*
 * Dequantize the given vertices.
 * Expects 3 components for both scale and offset.
 * Will write 3 * numVertices outputs to *v.
 */
void vkr_dequantize_vertices(
    const uint64_t *vq, const uint64_t numVertices, 
    const float *scale, const float *offset,
    float *v);

/*
 * Dequantize the given normals.
 * Will write 3 * numNormals outputs to *n and 2 * numNormals
 * outputs to *uv.
 */
void vkr_dequantize_normal_uv(
    const uint64_t *nq, const uint64_t numNormals, 
    float *n, float *uv);

/*
 * Convert the given texture into the .vkt format.
 * This function upsamples to the next power of two, creates mipmaps, and then
 * converts pixel data to the given output format.
 * The output file will always be in .vkt format.
 *
 * Note: Not all VkFormat values are supported!
 * Note: The error handler is optional.
 *
 * TODO: Document supported output formats
 * TODO: Make sure that hdr images can be converted (16bit/32bit per channel
 *                                                   RGB).
 * TODO (in usage): Everything BC1 except normals, which use BC5.
 */
VkrResult vkr_convert_texture(
    const char *inputFile, const char *outputFile,
    VkrTextureFormat outputFormat, VkrTextureFormat opaqueOutputFormat,
    VkrErrorHandler errorHandler);

#if defined(__cplusplus)
} // extern "C" {
#endif

