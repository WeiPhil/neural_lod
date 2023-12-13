#ifndef NEURAL_LOD_LEARNING_COMMON_CUH
#define NEURAL_LOD_LEARNING_COMMON_CUH

#include "neural_lod_learning.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>

using namespace neural_lod_learning;

inline __host__ __device__ float safe_acos(float x)
{
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return acos(x);
}

inline __host__ __device__ float safe_sqrt(float x)
{
    return sqrt(max(0.f,x));
}

inline __host__ __device__ glm::vec2 cartesian_to_spherical(glm::vec3 v)
{
    // Setting wo
    float theta = safe_acos(v.z);  // theta

    float phi = 0.0f;
    if (abs(v.x) > 0 || abs(v.y) > 0 ) {
        phi = atan2(v.y, v.x);
    } else {
        phi = 0.0;
    }
    return glm::vec2(theta, phi);
}

inline __host__ __device__ glm::vec3 spherical_to_cartesian(const float theta, const float phi)
{
    return glm::vec3(cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta));
}

inline __host__ __device__ glm::vec3 square_to_uniform_sphere(const glm::vec2 rng)
{
    float phi = 2.0f * M_PI * rng.x;
    float cos_theta = rng.y * 2.0f - 1.0f;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return glm::vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}


inline __host__ __device__ float rescale(const float value,
                                         const float old_scale_min,
                                         const float old_scale_max,
                                         const float new_scale_min,
                                         const float new_scale_max)
{
    return ((value - old_scale_min) / (old_scale_max - old_scale_min)) *
               (new_scale_max - new_scale_min) +
           new_scale_min;
}

inline __host__ __device__ float min_component(const glm::vec3 v)
{
    return glm::min(glm::min(v.x, v.y), v.z);
}

inline __host__ __device__ glm::u32 min_component(const glm::uvec3 v)
{
    return glm::min(glm::min(v.x, v.y), v.z);
}

inline __host__ __device__ float max_component(const glm::vec3 v)
{
    return glm::max(glm::max(v.x, v.y), v.z);
}

inline __host__ __device__ glm::u32 max_component(const glm::uvec3 v)
{
    return glm::max(glm::max(v.x, v.y), v.z);
}

inline __host__ __device__ glm::vec3 rescale(const glm::vec3 value,
                                         const glm::vec3 old_scale_min,
                                         const glm::vec3 old_scale_max,
                                         const glm::vec3 new_scale_min,
                                         const glm::vec3 new_scale_max)
{
    return glm::vec3(
        rescale(
            value[0], old_scale_min[0], old_scale_max[0], new_scale_min[0], new_scale_max[0]),
        rescale(
            value[1], old_scale_min[1], old_scale_max[1], new_scale_min[1], new_scale_max[1]),
        rescale(
            value[2], old_scale_min[2], old_scale_max[2], new_scale_min[2], new_scale_max[2]));
}


inline __host__ __device__ glm::vec3 jet_colormap(float v, float vmin = 0.0f,float vmax = 1.0f)
{  
   glm::vec3 c(1.0f);
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25f * dv)) {
      c.r = 0.f;
      c.g = 4.f * (v - vmin) / dv;
   } else if (v < (vmin + 0.5f * dv)) {
      c.r = 0.f;
      c.b = 1.f + 4.f * (vmin + 0.25f * dv - v) / dv;
   } else if (v < (vmin + 0.75f * dv)) {
      c.r = 4.f * (v - vmin - 0.5f * dv) / dv;
      c.b = 0.f;
   } else {
      c.g = 1.f + 4.f * (vmin + 0.75f * dv - v) / dv;
      c.b = 0;
   }

   return c;
}

inline __host__ __device__ int coord_6d_to_1d(
    int x, int y, int z, int a, int b, int c, int max_x, int max_y, int max_z, int max_a, int max_b, int max_c)
{
   int index = x + y * max_x + z * max_x * max_y + a * max_x * max_y * max_z +
               b * max_x * max_y * max_z * max_a + c * max_x * max_y * max_z * max_a * max_b;
   if (index >= max_x * max_y * max_z * max_a * max_b * max_c) {
      printf("x : %i/%i, y : %i/%i, z : %i/%i, a : %i/%i, b : %i/%i, c : %i/%i\n",
             x,
             max_x,
             y,
             max_y,
             z,
             max_z,
             a,
             max_a,
             b,
             max_b,
             c,
             max_c);
    }
    assert(x  < max_x);
    assert(y  < max_y);
    assert(z  < max_z);
    assert(a  < max_a);
    assert(b  < max_b);
    assert(c  < max_c);
    return index;
}

inline __host__ __device__ int coord_5d_to_1d(
    int x, int y, int z, int w, int s, int max_x, int max_y, int max_z, int max_w, int max_s)
{   
    int index = x + y * max_x + z * max_x * max_y + w * max_x * max_y * max_z + s * max_x * max_y * max_z * max_w;
    if(index  >= max_x * max_y * max_z * max_w * max_s){
      printf("x : %i/%i, y : %i/%i, z : %i/%i, w : %i/%i, s : %i/%i\n",
             x,
             max_x,
             y,
             max_y,
             z,
             max_z,
             w,
             max_w,
             s,
             max_s);
    }
    assert(x  < max_x);
    assert(y  < max_y);
    assert(z  < max_z);
    assert(w  < max_w);
    assert(s  < max_s);
    return index;
}

inline __host__ __device__ int coord_4d_to_1d(
    int x, int y, int z, int w, int max_x, int max_y, int max_z, int max_w)
{   
    int index = x + y * max_x + z * max_x * max_y + w * max_x * max_y * max_z;
    assert(index  < max_x * max_y * max_z * max_w);
    assert(x  < max_x);
    assert(y  < max_y);
    assert(z  < max_z);
    assert(w  < max_w);
    return index;
}

inline __host__ __device__ glm::ivec4 coord_1d_to_4d(
    int index, int max_x, int max_y, int max_z, int max_w)
{
    int x = index % max_x;
    int y = ( ( index - x ) / max_x ) %  max_y;
    int z = ( ( index - y * max_x - x ) / (max_x * max_y) ) % max_z; 
    int w = ( ( index - z * max_y * max_x - y * max_x - x ) / (max_x * max_y * max_z) ) % max_w; 
    assert(x < max_x);
    assert(y < max_y);
    assert(z < max_z);
    assert(w < max_w);
    return glm::ivec4(x,y,z,w);
}

inline __host__ __device__ int coord_3d_to_1d(
    int x, int y, int z, int max_x, int max_y, int max_z)
{   
    int index = x + y * max_x + z * max_x * max_y;
    assert(index  < max_x * max_y * max_z);
    assert(x  < max_x);
    assert(y  < max_y);
    assert(z  < max_z);
    return index;
}

inline __host__ __device__ glm::ivec3 coord_1d_to_3d(
    int index, int max_x, int max_y, int max_z)
{
    int x = index % max_x;
    int y = (index / max_x) % max_y;
    int z = index / (max_y * max_x);

    assert(x < max_x);
    assert(y < max_y);
    assert(z < max_z);
    return glm::ivec3(x,y,z);
}

inline __host__ __device__ int coord_2d_to_1d(
    int x, int y, int max_x, int max_y)
{   
    int index = x + y * max_x;
    assert(index < max_x * max_y);
    assert(x  < max_x);
    assert(y  < max_y);
    return index;
}

inline __host__ __device__ glm::ivec2 coord_1d_to_2d(int index, int max_x, int max_y)
{
    int x = index % max_x;
    int y = (index / max_x ) %  max_y;
    assert(x < max_x);
    assert(y < max_y);
    return glm::ivec2(x,y);
}

inline __host__ __device__ void get_deterministic_local_pi_po(
    const glm::vec3 input_local_pi_po[2],
    glm::vec3 &output_local_pi,
    glm::vec3 &output_local_po)
{
    NeuralLodLearning::CubeFace faces[2];
    for (size_t j = 0; j < 2; j++)
    {
        // find the largest axis to determine face
        float largest_dist = -2e20;
        for (size_t i = 0; i < 3; i++)
        {
            float dist = abs(input_local_pi_po[j][i]);
            if( dist > largest_dist){
                faces[j] = (NeuralLodLearning::CubeFace)(input_local_pi_po[j][i] > 0.0 ? i * 2 : i * 2 + 1);
                largest_dist = dist;
            }
        } 
    }

    const int deterministic_face_pairs[15][2] = {{0, 1},
                                                 {0, 2},
                                                 {0, 3},
                                                 {0, 4},
                                                 {0, 5},
                                                 {1, 2},
                                                 {1, 3},
                                                 {1, 4},
                                                 {1, 5},
                                                 {2, 3},
                                                 {2, 4},
                                                 {2, 5},
                                                 {3, 4},
                                                 {3, 5},
                                                 {4, 5}};

    // Now find the right output for the deterministic po pi
    for (size_t i = 0; i < 15; i++)
    {
        if (deterministic_face_pairs[i][0] == faces[0] &&
            deterministic_face_pairs[i][1] == faces[1]) {
            // unchanged order
            output_local_pi = input_local_pi_po[0];
            output_local_po = input_local_pi_po[1];
            return;
        } else if (deterministic_face_pairs[i][0] == faces[1] &&
                    deterministic_face_pairs[i][1] == faces[0]){
            // exchange order
            output_local_pi = input_local_pi_po[1];
            output_local_po = input_local_pi_po[0];
            return;
        }
    }
}

inline __host__ __device__ glm::vec3 sample_cube_face(const glm::vec3 rng, glm::vec3& face_normal)
{
    int index = glm::min(int(rng.x * 6),5);
    assert(index < 6 && index >= 0);
    
    glm::vec3 cube_pos;

    float u = rng.y - 0.5f;
    float v = rng.z - 0.5f;

    switch (index) {
    case NeuralLodLearning::CubeFace::XPlus:
        cube_pos = glm::vec3(0.5f, u, v);
        face_normal = glm::vec3(1.f, 0.f, 0.f);
        break;
    case NeuralLodLearning::CubeFace::XMinus:
        cube_pos = glm::vec3(-0.5f, u, v);
        face_normal = glm::vec3(-1.f, 0.f, 0.f);
        break;
    case NeuralLodLearning::CubeFace::YPlus:
        cube_pos = glm::vec3(u, 0.5f, v);
        face_normal = glm::vec3(0.f, 1.f, 0.f);
        break;
    case NeuralLodLearning::CubeFace::YMinus:
        cube_pos = glm::vec3(u, -0.5f, v);
        face_normal = glm::vec3(0.f, -1.f, 0.f);
        break;
    case NeuralLodLearning::CubeFace::ZPlus:
        cube_pos = glm::vec3(u, v, 0.5f);
        face_normal = glm::vec3(0.f, 0.f, 1.f);
        break;
    case NeuralLodLearning::CubeFace::ZMinus:
        cube_pos = glm::vec3(u, v, -0.5f);
        face_normal = glm::vec3(0.f, 0.f, -1.f);
        break;
    default:
        assert(false);
        break;
    }
    return cube_pos;
}

inline __host__ __device__ glm::vec3 local_voxel_pos_from_uv_and_face(glm::vec2 uv, NeuralLodLearning::CubeFace face)
{
    glm::vec3 cube_pos;

    uv -= glm::vec2(0.5f);

    switch (face) {
    case NeuralLodLearning::CubeFace::XPlus:
        cube_pos = glm::vec3(0.5f, uv.x, uv.y);
        break;
    case NeuralLodLearning::CubeFace::XMinus:
        cube_pos =  glm::vec3(-0.5f, uv.x, uv.y);
        break;
    case NeuralLodLearning::CubeFace::YPlus:
        cube_pos = glm::vec3(uv.x, 0.5f, uv.y);
        break;
    case NeuralLodLearning::CubeFace::YMinus:
        cube_pos = glm::vec3(uv.x, -0.5f, uv.y);
        break;
    case NeuralLodLearning::CubeFace::ZPlus:
        cube_pos = glm::vec3(uv.x, uv.y, 0.5f);
        break;
    case NeuralLodLearning::CubeFace::ZMinus:
        cube_pos = glm::vec3(uv.x, uv.y, -0.5f);
        break;
    default:
        assert(false);
    }
    return cube_pos;
}

inline __host__ __device__ glm::vec2 uv_and_face_from_local_voxel_pos(const glm::vec3 local_pos, NeuralLodLearning::CubeFace& face)
{
    if (local_pos.x < -(0.5+1e-5) || local_pos.y < -(0.5+1e-5) || local_pos.z < -(0.5+1e-5) || local_pos.x > (0.5+1e-5) ||
        local_pos.y > (0.5+1e-5) || local_pos.z > (0.5+1e-5)) {
        printf("local_pos not in [-0.5,0.5] : %f %f %f\n", local_pos.x, local_pos.y, local_pos.z);
    }
    float largest_dist = -2e20;
    // find the largest axis
    for (size_t i = 0; i < 3; i++)
    {
        float dist = abs(local_pos[i]);
        if( dist > largest_dist){
            face = (NeuralLodLearning::CubeFace)(local_pos[i] > 0.0 ? i * 2 : i * 2 + 1);
            largest_dist = dist;
        }
    }

    int dim = face / 2;
    glm::vec2 uv;
    if(dim == 0){
        uv = glm::vec2(local_pos.y,local_pos.z);
    }else if(dim == 1){
        uv = glm::vec2(local_pos.x,local_pos.z); 
    }else if(dim == 2){
        uv = glm::vec2(local_pos.x,local_pos.y);
    }

    uv += glm::vec2(0.5f);

    return uv;
}

template<typename T>
inline __device__ glm::vec3 nanovdb_to_vec3(nanovdb::Vec3<T> nanovdb_vec){
    return glm::vec3(nanovdb_vec[0],nanovdb_vec[1],nanovdb_vec[2]);
}

template<typename T>
inline __device__ nanovdb::Vec3<T> vec3_to_nanovdb(glm::vec3 glm_vec){
    return nanovdb::Vec3<T>(glm_vec[0],glm_vec[1],glm_vec[2]);
}

inline __device__ nanovdb::Coord ivec3_to_coord(glm::ivec3 glm_vec){
    return nanovdb::Coord(glm_vec[0],glm_vec[1],glm_vec[2]);
}

inline __device__ glm::ivec3 coord_to_ivec3(nanovdb::Coord nanovdb_coord){
    return glm::ivec3(nanovdb_coord[0],nanovdb_coord[1],nanovdb_coord[2]);
}

inline __host__ __device__ glm::vec3 world_to_local_voxel_pos(
    const glm::vec3 &world_pos, const NeuralLodLearning::Voxel &voxel, bool clamp_to_range = true)
{
    // reduces to -0.5,0.5 and then map to 0,1
    glm::vec3 pos = (world_pos - voxel.center) / voxel.extent;

    // Alert only for large epsilon due to transform inacuracies in grid intersection
    if(pos.x < -0.5002 || pos.x > 0.5002 || pos.y < -0.5002 || pos.y > 0.5002 || pos.z < -0.5002 || pos.z > 0.5002){
        printf("pos not in [-0.5002,0.5002] %f %f %f\n",pos.x,pos.y,pos.z);
    }
    if(clamp_to_range){
        pos = glm::clamp(pos, glm::vec3(-0.5f), glm::vec3(0.5f));
    }
    return pos;
}

inline __host__ __device__ glm::vec3 index_to_local_voxel_pos(
    const glm::vec3 &index_pos, const glm::ivec3 &lod_voxel_idx, bool clamp_to_range = true)
{
    // reduces to -0.5,0.5 and then map to 0,1
    glm::vec3 pos = (index_pos - glm::vec3(lod_voxel_idx)) - 0.5f;

    // Alert only for large epsilon due to transform inacuracies in grid intersection
    if(pos.x < -0.501 || pos.x > 0.501 || pos.y < -0.501 || pos.y > 0.501 || pos.z < -0.501 || pos.z > 0.501){
        printf("pos not in [-0.501,0.501] %f %f %f\n",pos.x,pos.y,pos.z);
    }
    if(clamp_to_range){
        pos = glm::clamp(pos, glm::vec3(-0.5f), glm::vec3(0.5f));
    }
    return pos;
}

inline __host__ __device__ uint32_t compute_inference_idx_for_throughput(
    glm::vec3 wo, glm::vec3 wi, glm::uvec2 sampling_res, uint32_t output_dims)
{
    glm::vec2 wo_theta_phi = cartesian_to_spherical(wo);
    glm::vec2 wi_theta_phi = cartesian_to_spherical(wi);

    int theta_wo_idx = min(int((wo_theta_phi.x / M_PI) * sampling_res.x), sampling_res.x - 1);
    int phi_wo_idx = min(int(((wo_theta_phi.y + M_PI) / (2.f * M_PI)) * sampling_res.y),
                         sampling_res.y - 1);
    int theta_wi_idx = min(int((wi_theta_phi.x / M_PI) * sampling_res.x), sampling_res.x - 1);
    int phi_wi_idx = min(int(((wi_theta_phi.y + M_PI) / (2.f * M_PI)) * sampling_res.y),
                         sampling_res.y - 1);

    int index = coord_4d_to_1d(theta_wo_idx,
                               phi_wo_idx,
                               theta_wi_idx,
                               phi_wi_idx,
                               sampling_res.x,
                               sampling_res.y,
                               sampling_res.x,
                               sampling_res.y);

    return index * output_dims;
}

// local pi/po assumed in [-0.5,0.5]
inline __host__ __device__ uint32_t
compute_inference_idx_for_visibility(glm::vec3 local_pi,
                                     glm::vec3 local_po,
                                     glm::uvec3 sampling_res,
                                     uint32_t output_dims)
{

    NeuralLodLearning::CubeFace pi_face_idx;
    glm::vec2 pi_uv = uv_and_face_from_local_voxel_pos(local_pi, pi_face_idx);

    int pi_idx_u = min(int(pi_uv.x * sampling_res.x), sampling_res.x - 1);
    int pi_idx_v = min(int(pi_uv.y * sampling_res.y), sampling_res.y - 1);

    NeuralLodLearning::CubeFace po_face_idx;
    glm::vec2 po_uv = uv_and_face_from_local_voxel_pos(local_po, po_face_idx);

    int po_idx_u = min(int(po_uv.x * sampling_res.x), sampling_res.x - 1);
    int po_idx_v = min(int(po_uv.y * sampling_res.y), sampling_res.y - 1);

    int index = coord_6d_to_1d(pi_idx_u,
                               pi_idx_v,
                               (int)pi_face_idx,
                               po_idx_u,
                               po_idx_v,
                               (int)po_face_idx,
                               sampling_res.x,
                               sampling_res.y,
                               sampling_res.z,
                               sampling_res.x,
                               sampling_res.y,
                               sampling_res.z);

    return index * output_dims;
}

inline __device__ bool intersect_sphere(glm::vec3 ray_origin,
                                        glm::vec3 ray_dir,
                                        glm::vec3 sphere_center,
                                        float radius,
                                        float &nearT,
                                        float &farT)
{
    glm::vec3 oc = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    // introducing 2h = b for optimization
    float half_b = dot(oc, ray_dir);
    float c = dot(oc, oc) - radius * radius;
    float delta = half_b * half_b - a * c;

    nearT = -1e20;
    farT  = 1e20;

    if (delta > 0.0) {
        float root = sqrt(delta);

        // the possible intersection on one side of the sphere
        float t1 = (-half_b - root) / a;
        float t2 = (-half_b + root) / a;

        nearT = std::max(t1, nearT);
        farT = std::min(t2, farT);

        if (!(nearT < farT) || farT < 0.0)
            return false;

        return true;
    } else if (delta == 0) {
        nearT = farT = -half_b;

        return true;
    } else {
        
        return false;
    }
}

inline __host__ __device__ bool intersect_aabb_min_max(glm::vec3 ray_origin,
                                          glm::vec3 ray_dir,
                                          glm::vec3 box_min,
                                          glm::vec3 box_max,
                                          float &nearT,
                                          float &farT)
{
    glm::vec3 ray_dir_inv = glm::vec3(1.f) / ray_dir;
    nearT = -1e20;
    farT  = 1e20;

    /* For each pair of AABB planes */
    for (int i=0; i< 3; i++) {
        const float origin = ray_origin[i];
        const float minVal = box_min[i], maxVal = box_max[i];

        if (ray_dir[i] == 0) {
            /* The ray is parallel to the planes */
            if (origin < minVal || origin > maxVal)
                return false;
        } else {
            /* Calculate intersection distances */
            float t1 = (minVal - origin) * ray_dir_inv[i];
            float t2 = (maxVal - origin) * ray_dir_inv[i];

            if (t1 > t2){
                float tmp = t1;
                t1 = t2;
                t2 = tmp;
            }

            nearT = std::max(t1, nearT);
            farT = std::min(t2, farT);

            if (!(nearT < farT) || farT < 0.0)
                return false;
        }
    }
    return true;
};

inline __device__ bool intersect_aabb(glm::vec3 ray_origin,
                                          glm::vec3 ray_dir,
                                          glm::vec3 box_center,
                                          float box_extent,
                                          float &nearT,
                                          float &farT)
{
    glm::vec3 box_min = box_center - 0.5f * box_extent;
    glm::vec3 box_max = box_center + 0.5f * box_extent;

    return intersect_aabb_min_max(ray_origin,ray_dir,box_min,box_max,nearT,farT);
};

template<typename RealT, typename RayT>
inline __device__ bool ray_intersect_sphere(const RayT& ray, const nanovdb::Vec3<RealT>& center, const RealT radius, RealT& t0)
{
    const bool use_robust_method = false;
    using namespace nanovdb;
    using Vec3 = Vec3<RealT>;

    const Vec3& rd = ray.dir();
    const Vec3& rd_inv = ray.invDir();
    const Vec3  ro = ray.eye() - center;

    RealT b = ro.dot(rd);
    RealT c = ro.dot(ro) - radius * radius;
    RealT disc = b * b - c;
    if (disc > 0.0f) {
        RealT sdisc = sqrtf(disc);
        RealT root1 = (-b - sdisc);

        bool do_refine = false;

        RealT root11 = 0.0f;

        if (use_robust_method && Abs(root1) > 10.f * radius) {
            do_refine = true;
        }

        if (do_refine) {
            // refine root1
            auto ro1 = ro + root1 * rd;
            b = ro1.dot(rd);
            c = ro1.dot(ro1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f) {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        bool check_second = true;

        RealT t;
        t = (root1 + root11);
        if (t > t0) {
            // normal = (ro + (root1 + root11) * rd) / radius;
            t0 = t;
            return true;
        }

        if (check_second) {
            RealT root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2;
            // normal = (ro + root2 * rd) / radius;
            if (t > t0) {
                t0 = t;
                return true;
            }
        }
    }

    return false;
}

template<typename RealT>
inline __device__ bool ray_intersect_cube(const nanovdb::Ray<RealT> &ray,
                                          const nanovdb::Vec3<RealT> &center,
                                          RealT radius,
                                          RealT &t)
{
    using namespace nanovdb;

    using Vec3 = Vec3<RealT>;

    const Vec3 &rd = ray.dir();
    const Vec3& rd_inv = ray.invDir();
    const Vec3  ro = ray.eye() - center;

    Vec3 sgn = -Vec3(nanovdb::Sign(rd[0]), nanovdb::Sign(rd[1]), nanovdb::Sign(rd[2]));
    Vec3 distanceToPlane = radius * sgn - ro;
    distanceToPlane = distanceToPlane * rd_inv;
    Vec3i test = Vec3i((distanceToPlane[0] >= 0.0f) &&
                           ((Abs(ro[1] + rd[1] * distanceToPlane[0]) < radius) && (Abs(ro[2] + rd[2] * distanceToPlane[0]) < radius)),
                       (distanceToPlane[1] >= 0.0f) &&
                           ((Abs(ro[2] + rd[2] * distanceToPlane[1]) < radius) && (Abs(ro[0] + rd[0] * distanceToPlane[1]) < radius)),
                       (distanceToPlane[2] >= 0.0f) &&
                           ((Abs(ro[0] + rd[0] * distanceToPlane[2]) < radius) && (Abs(ro[1] + rd[1] * distanceToPlane[2]) < radius)));

    sgn = test[0] ? Vec3(sgn[0], 0.0f, 0.0f)
                  : (test[1] ? Vec3(0.0f, sgn[1], 0.0f) : Vec3(0.0f, 0.0f, test[2] ? sgn[2] : 0.0f));
    t = (sgn[0] != 0.0f) ? distanceToPlane[0] : ((sgn[1] != 0.0) ? distanceToPlane[1] : distanceToPlane[2]);
    // normal = sgn;
    return (sgn[0] != 0) || (sgn[1] != 0) || (sgn[2] != 0);
}

template<typename RealT>
inline __device__ bool ray_intersect_voxel(int voxelGeometry,
                                            const nanovdb::Ray<RealT> &ray,
                                            const nanovdb::Vec3<RealT> &center,
                                            RealT radius,
                                            RealT &t)
{
    if (voxelGeometry == 1)
        return ray_intersect_sphere(ray, center, radius, t);
    return ray_intersect_cube(ray, center, radius, t);
}

inline __host__ __device__ uint32_t compute_inference_sample_count_lod_offset(uint32_t lod, uint32_t min_lod = 1)
{
    uint32_t lod_offset = 0;
    for(int i = min_lod; i < lod; i++)
        lod_offset += 1 << ((VOXEL_GRID_MAX_POWER_2()- i) * 3);
    return lod_offset;
};

template <typename RealT, typename RayT, typename AccT>
inline __device__ bool accumulate_samples_along_ray(
    RayT &iRay,
    AccT &acc,
    nanovdb::Coord &ijk,
    RealT &t_start,
    RealT &t_end,
    uint32_t *inferences_sample_count_grid,
    uint32_t current_lod,
    uint32_t &accumulated)
{
    using TreeT = nanovdb::NanoTree<float>;

    nanovdb::TreeMarcher<TreeT::LeafNodeType, RayT, AccT> marcher(acc);
    if (marcher.init(iRay)) {

        const TreeT::LeafNodeType *node = nullptr;
        float t0 = 0, t1 = 0;

        while (marcher.step(&node, t0, t1)) {
            nanovdb::DDA<RayT> dda;
            dda.init(marcher.ray(), t0, t1);
            do {
                ijk = dda.voxel();
                // CAVEAT:
                // This is currently necessary becuse the voxel returned might not actually be innside the node!
                // This is currently happening from time to time due to float precision issues,
                // so we skip out of bounds voxels here...
                auto localIjk = ijk - node->origin();
                if (localIjk[0] < 0 || localIjk[1] < 0 || localIjk[2] < 0 || localIjk[0] >= 8 || localIjk[1] >= 8 || localIjk[2] >= 8)
                    continue;

                const uint32_t offset = node->CoordToOffset(ijk);
                if (node->isActive(offset)){
                    glm::uvec3 voxel_idx = glm::uvec3(coord_to_ivec3(ijk));
                    uint32_t offset = compute_inference_sample_count_lod_offset(current_lod);
                    accumulated +=
                        inferences_sample_count_grid[offset + encode_morton3(voxel_idx.x,
                                                                         voxel_idx.y,
                                                                         voxel_idx.z)];
                }

            } while (dda.step());
        }
    }
    return false;
}

template <typename RealT, typename RayT, typename AccT>
inline __device__ bool first_active_voxel_intersection(
    RayT &iRay,
    AccT &acc,
    nanovdb::Coord &ijk,
    RealT &t_start,
    RealT &t_end,
    float &grid_value,
    const nanovdb::Coord previous_ijk = nanovdb::Coord(-1))
{
    using TreeT = nanovdb::NanoTree<float>;

    nanovdb::TreeMarcher<TreeT::LeafNodeType, RayT, AccT> marcher(acc);
    if (marcher.init(iRay)) {

        const TreeT::LeafNodeType *node = nullptr;
        float t0 = 0, t1 = 0;

        while (marcher.step(&node, t0, t1)) {
            nanovdb::DDA<RayT> dda;
            dda.init(marcher.ray(), t0, t1);
            do {
                ijk = dda.voxel();
                // CAVEAT:
                // This is currently necessary becuse the voxel returned might not actually be innside the node!
                // This is currently happening from time to time due to float precision issues,
                // so we skip out of bounds voxels here...
                auto localIjk = ijk - node->origin();
                if (localIjk[0] < 0 || localIjk[1] < 0 || localIjk[2] < 0 || localIjk[0] >= 8 || localIjk[1] >= 8 || localIjk[2] >= 8)
                    continue;

                const uint32_t offset = node->CoordToOffset(ijk);
                if (node->isActive(offset) && ijk != previous_ijk){
                    glm::vec3 ray_origin = nanovdb_to_vec3(marcher.ray().eye());
                    glm::vec3 ray_dir = nanovdb_to_vec3(marcher.ray().dir());
                    glm::vec3 voxel_min = coord_to_ivec3(ijk);
                    float t_near, t_far;
                    if(intersect_aabb_min_max(ray_origin, ray_dir, voxel_min, voxel_min + 1.0f, t_near, t_far)){
                        t_start = t_near;//dda.time();
                        t_end = t_far; //dda.next();
                        grid_value = node->getValue(offset);
                        return true;
                    }
                }

            } while (dda.step());
        }
    }
    return false;
}

/// ************ Metrics **************///

inline __host__ __device__ float mcc(uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp)
{
    float numer = float(tp * tn) - float(fp * fn);
    float denom_sqr = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn);
    return denom_sqr == 0 ? 0.f : numer / sqrtf(denom_sqr);
}

inline __host__ __device__ float precision_score(uint32_t fp, uint32_t tp)
{
    return tp == 0 ? 0.f : float(tp) / float(tp + fp);
}
inline __host__ __device__ float recall_score(uint32_t fn, uint32_t tp)
{
    return tp == 0 ? 0.f : float(tp) / float(tp + fn);
}

inline __host__ __device__ float fscore(
    uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp, float beta)
{
    float p = precision_score(fp, tp);
    float r = recall_score(fn, tp);

    if (p + r == 0.f)
        return 0.f;

    const float beta_sqr = beta * beta;
    return ((1.f + beta_sqr) * p * r) / (beta_sqr * p + r);
}

inline __host__ __device__ float fscore_macro(
    uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp, float beta)
{
    float f_pos = fscore(tn,fp,fn,tp,beta);
    float f_neg = fscore(tp,fn,fp,tn,beta);
    return (f_pos + f_neg)*0.5f;
}

inline __host__ __device__ float fscore_weighted(
    uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp, glm::uvec2 support, float beta)
{
    float f_pos = fscore(tn, fp, fn, tp, beta);
    float f_neg = fscore(tp, fn, fp, tn, beta);
    return (f_pos * support[0] + f_neg * support[1]) / (support[0] + support[1]);
}

inline __host__ __device__ float optimal_fscore_weighted(
    uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp, glm::uvec2 support, float beta)
{
    float f_pos, f_neg;
    f_pos = fscore(tn, fp, fn, tp, beta);
    f_neg = fscore(tp, fn, fp, tn, 1.f);
    return (f_pos * support[0] + f_neg * support[1]) / (support[0] + support[1]);
}


inline __host__ __device__ float cohen_kappa(uint32_t tn,
                                             uint32_t fp,
                                             uint32_t fn,
                                             uint32_t tp)
{
    float numer = 2.f * (float(tp * tn) - float(fn * fp));
    float denom_sqr = float(tp + fp) * float(fp + tn) + float(tp + fn) * float(fn + tn);
    return denom_sqr == 0 ? 0.f : numer / denom_sqr;
}

inline __host__ __device__ float accuracy(uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp)
{
    float numer = float(tp + tn);
    float denom_sqr = float(tp + tn + fp + fn);
    return denom_sqr == 0 ? 0.f : numer / denom_sqr;
}

inline __host__ __device__ float balanced_accuracy(uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp)
{
    float sensitivity = tp == 0 ? 0.0f : float(tp) / float(tp + fn);
    float specificity = tn == 0 ? 0.0f : float(tn) / float(tn + fp);
    return (sensitivity + specificity) * 0.5f;
}

inline __host__ __device__ float fowlkes_mallows(uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp)
{
    float p = precision_score(fp, tp);
    float r = recall_score(fn, tp);
    return sqrtf(p*r);
}

inline __host__ __device__ float youden_j(uint32_t tn, uint32_t fp, uint32_t fn, uint32_t tp)
{
    float sensitivity = tp == 0 ? 0.0f : float(tp) / float(tp + fn);
    float specificity = tn == 0 ? 0.0f : float(tn) / float(tn + fp);
    return sensitivity + specificity - 1.f;
}

#endif //NEURAL_LOD_LEARNING_H
