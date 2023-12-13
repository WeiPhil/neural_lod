#ifndef NEURAL_LOD_LEARNING_GRID_CUH
#define NEURAL_LOD_LEARNING_GRID_CUH

#include <tiny-cuda-nn/common.h>

#include "neural_lod_learning.h"

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN


inline __host__ __device__ uint32_t compute_voxel_grid_lod_byte_offset(uint32_t lod)
{
    uint32_t lod_byte_offset = 0;
    for(int i = 0; i < lod; i++)
        lod_byte_offset += 1 << ((VOXEL_GRID_MAX_POWER_2() - i) * 3 - 3);
    return lod_byte_offset;
};

inline __device__ uint32_t compute_voxel_grid_lod_morton_idx(uint32_t lod_0_morton_idx, uint32_t lod)
{
    uint32_t morton_idx = lod_0_morton_idx;
    if (lod > 0) {
        glm::uvec3 xyz = decode_morton3(lod_0_morton_idx);
        xyz.x >>= lod;
        xyz.y >>= lod;
        xyz.z >>= lod;
        morton_idx = encode_morton3(xyz.x,xyz.y,xyz.z);
    }
    return morton_idx;
};

inline __device__ bool clip_index_ray(glm::vec3 &index_origin,
                                      const glm::vec3 index_dir,
                                      uint32_t res)
{
    float t_near, t_far;
    if (!intersect_aabb_min_max(
            index_origin, index_dir, glm::vec3(0.f), glm::vec3(res), t_near, t_far)) {
        return false;
    }
    const float EPS = 1e-4;
    // If we are outside of the bounding box clip the origin the origin
    index_origin = t_near > 0 ? glm::clamp(index_origin + t_near * index_dir,
                                           glm::vec3(EPS),
                                           glm::vec3(res - EPS))
                              : index_origin;
    return true;
}

inline __host__ __device__ NeuralLodLearning::Voxel compute_lod_voxel(
    glm::uvec3 lod_voxel_idx, // The xyz coordinate at the current lod
    const NeuralLodLearning::AABox<glm::vec3> &voxel_grid_aabb,
    float voxel_size, // Voxel size at lod 0
    int lod_level)
{
    float step_size = scalbn(voxel_size,lod_level);

    if (lod_voxel_idx.x >= (VOXEL_GRID_MAX_RES() >> lod_level) ||
        lod_voxel_idx.y >= (VOXEL_GRID_MAX_RES() >> lod_level) ||
        lod_voxel_idx.z >= (VOXEL_GRID_MAX_RES() >> lod_level)) {
        printf("lod_level : %i lod_voxel_idx : %i %i %i grid_max_res : %i\n",
               lod_level,
               lod_voxel_idx.x,
               lod_voxel_idx.y,
               lod_voxel_idx.z,
               VOXEL_GRID_MAX_RES());
        assert(false);
    }

    glm::vec3 voxel_min = voxel_grid_aabb.min + glm::vec3(lod_voxel_idx) * glm::vec3(step_size);

    NeuralLodLearning::Voxel voxel;
    voxel.center = voxel_min + glm::vec3(step_size * 0.5f);
    voxel.extent = step_size;

    return voxel;
}


inline __host__ __device__ NeuralLodLearning::Voxel compute_voxel(
    int voxel_idx,
    const NeuralLodLearning::AABox<glm::vec3> &voxel_grid_aabb,
    float voxel_size,
    uint32_t lod_level)
{
    glm::ivec3 lod_xyz = coord_1d_to_3d(voxel_idx, VOXEL_GRID_MAX_RES(),VOXEL_GRID_MAX_RES(),VOXEL_GRID_MAX_RES());
    lod_xyz.x >>= lod_level;
    lod_xyz.y >>= lod_level;
    lod_xyz.z >>= lod_level;
    return compute_lod_voxel(lod_xyz,voxel_grid_aabb,voxel_size,lod_level);
}


NEURAL_LOD_LEARNING_NAMESPACE_END

#endif