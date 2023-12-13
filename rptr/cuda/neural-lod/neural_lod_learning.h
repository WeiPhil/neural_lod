#pragma once
#ifndef NEURAL_LOD_LEARNING_H
#define NEURAL_LOD_LEARNING_H

#include "../render_cuda.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#ifdef TCNN_NAMESPACE_BEGIN
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#endif

#include <fstream>
#include <iostream>

// Comment to disable aabb padding
#define VOXELIZER_REGULARISATION

inline constexpr __device__ __host__ uint32_t VOXEL_GRID_MAX_POWER_2() {
	return 9;
}

inline constexpr __device__ __host__ uint32_t NUM_LODS() {
	return 7;
}

inline constexpr __device__ __host__ uint32_t VOXEL_GRID_MAX_RES() {
	return 1 << VOXEL_GRID_MAX_POWER_2();
}

inline constexpr __device__ __host__ uint32_t VOXEL_GRID_MIN_RES() {
	return VOXEL_GRID_MAX_RES() >> (NUM_LODS() - 1);
}

// Morton encoding would be lossy at that resolution!
static_assert(VOXEL_GRID_MAX_RES() <= 1024, "static_assert");
// We need at least one  encoding would be lossy at that resolution!
static_assert((int) VOXEL_GRID_MAX_POWER_2() - (int) NUM_LODS() >= 0, "static_assert");

#define MAX_FILENAME_SIZE 512

#include "optim_config.h"

constexpr const char *lod_levels[] = {"LoD 0", "LoD 1", "LoD 2", "LoD 3", "LoD 4", "LoD 5", "LoD 6", "LoD 7", "LoD 8"};

// Forward definition instead of "util/interactive_camera.h" include because of some glm include clash
struct OrientedCamera;

#define NEURAL_LOD_LEARNING_NAMESPACE_BEGIN namespace neural_lod_learning {
#define NEURAL_LOD_LEARNING_NAMESPACE_END }

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN

#define ARRAYSIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*(_ARR))))

using namespace optimconfig;

constexpr const char *debug_views[] = {"Voxel Grid", "Throughput", "Visibility"};

constexpr const char *threshold_optim_metrics[] = {"Precision", "Recall", "FScore", "FScoreMacro","FScoreWeighted","OptimalFScoreWeighted", "MCC", "Cohen's Kappa", "Accuracy","BalancedAccuracy", "Fowlkes-Mallows","Youden-J Statistic"};

const uint32_t throughput_n_input_dims = 9;   // (voxel_center_position, wo (towards camera), wi)
const uint32_t throughput_n_output_dims = 3;  // RGB throughput
const uint32_t visibility_n_input_dims = 6; // (pi, po)

const uint32_t visibility_n_output_dims = 1;  // Visibility mask
const uint32_t ray_query_count = DEFAULT_RAY_QUERY_BUDGET; // (needs to be smaller than DEFAULT_RAY_QUERY_BUDGET)

// Helper for many tiny-cuda-nn objects
template <class T, class X>
inline T &reinitialize_object(T &obj, X &&other)
{
    obj.~T();
    new (&obj) T((X &&) other);
    return obj;
}

struct NeuralLodLearning : RenderExtension {

    enum CubeFace {
        XPlus  = 0,
        XMinus = 1,
        YPlus = 2,
        YMinus = 3,
        ZPlus = 4,
        ZMinus = 5,
        SIZE = 6,
    };

    enum OptimisationType {
        ThroughputOptim,
        VisibilityOptim,
        ThresholdsOptim,
    };

    enum DebugView {
        VoxelGridView,
        ThroughputView,
    };
    
    enum ThresholdOptimMetrics {
        Precision,
        Recall,
        FScore,
        FScoreMacro,
        FScoreWeighted,
        OptimalFScoreWeighted,
        MCC,
        CohenKappa,
        Accuracy,
        BalancedAccuracy,
        FowlkesMallows,
        YoudenJ
    };

    struct ThroughputNNParams {

        glm::uvec2 inference_2d_sampling_res = glm::uvec2(24, 24);

        // Voxel sampling distribution (default uniform)
        float pdf_strength = 0.f;
        float pdf_shift = 0.f;

        // Other optimisation options
        bool log_learning;

        // Various constants for the network and optimization
        uint32_t batch_size = ray_query_count;
        uint32_t n_training_steps = 30000;
        // Optimizer
        OptimizerType optimizer = OptimizerType::Adam;
        OptimizerOptions optimizer_options;
        // Loss
        LossType loss = LossType::RelativeL2;
        // Network
        NetworkType network = NetworkType::FullyFusedMLP;
        ActivationType activation = ActivationType::ReLU;
        ActivationType output_activation = ActivationType::None;
        int n_neurons = 128;
        int n_hidden_layers = 4;        
        // Encoding
        EncodingType voxel_encoding = EncodingType::HashGrid;
        GridOptions voxel_grid_options;
        // Outgoing Direction Encoding
        EncodingType outgoing_encoding = EncodingType::Identity;
        GridOptions outgoing_grid_options;
        ShOptions outgoing_sh_options;
        // Incident Direction Encoding
        EncodingType incident_encoding = EncodingType::Identity;
        GridOptions incident_grid_options;
        ShOptions incident_sh_options;
    };

    struct VisibilityNNParams {
        glm::uvec3 inference_3d_sampling_res = glm::uvec3(16, 16, 6);

        // Voxel sampling distribution (default uniform)
        float pdf_strength = 0.f;
        float pdf_shift = 0.f;

        bool increase_boundary_density = false;
        float voxel_extent_dilation_outward = 0.005f;
        float voxel_extent_dilation_inward = 0.f;
        float voxel_extent_min_dilation = 1e-4f;
        float voxel_bound_bias = 0.f;

        // Various constants for the network and optimization
        uint32_t batch_size = ray_query_count;
        uint32_t n_training_steps = 30000;
        // Optimizer
        OptimizerType optimizer = OptimizerType::Adam;
        OptimizerOptions optimizer_options;
        // Loss
        LossType loss = LossType::BinaryCrossEntropy;
        // Network
        NetworkType network = NetworkType::FullyFusedMLP;
        ActivationType activation = ActivationType::ReLU;
        ActivationType output_activation = ActivationType::Sigmoid;
        int n_neurons = 128;
        int n_hidden_layers = 4;
        // Voxel Center Encoding
        EncodingType voxel_encoding = EncodingType::HashGrid;
        GridOptions voxel_grid_options;
        // Direction Encoding
        EncodingType voxel_entry_exit_encoding = EncodingType::Identity;
        GridOptions voxel_entry_exit_grid_options;
        // Position Encoding
        EncodingType position_encoding = EncodingType::Identity;
        GridOptions position_grid_options;
    };

    struct NeuralLodLearningParams {
        ThroughputNNParams throughput_nn;
        VisibilityNNParams visibility_nn;

        // Debug view
        bool square_debug_view = false;
        DebugView debug_view = DebugView::VoxelGridView;
    
        // Optimisiation params
        bool run_optimisation = false;
        bool compute_inference_during_training = true;
        OptimisationType optimisation_type = OptimisationType::ThroughputOptim;
        glm::vec2 theta_phi_fixed = glm::vec2(0.f,0.f);
        glm::ivec3 selected_voxel_idx = glm::ivec3(0,0,0);
        int last_sparse_idx = 0;
        bool compute_optimisation_ref = false;
        bool show_optimisation_ref = true;
        bool learn_single_voxel = false;
        int max_throughput_depth = 1000;
        bool debug_grid_data = false;

        // Threshold optim params
        uint32_t threshold_optim_samples = 1000;
        uint32_t threshold_max_parralel_voxels = 1000;
        uint32_t threshold_min_lod = 0;
        uint32_t threshold_max_lod = NUM_LODS();
        float threshold_fixed_value = 0.5f;
        float threshold_min_value = 0.1f;
        float threshold_max_value = 0.9f;
        ThresholdOptimMetrics threshold_optim_metric = ThresholdOptimMetrics::OptimalFScoreWeighted;
        float threshold_fscore_beta = 1.f;
    };

    template <typename T>
    struct AABox {
        T min;
        T max;
        inline __host__ __device__ T extent() const { return max - min;}

        inline __host__ __device__ float smallest_distance_from(const glm::vec3 &point)
        {
            float dx = std::max(std::max(min.x - point.x, point.x - max.x), 0.0f);
            float dy = std::max(std::max(min.y - point.y, point.y - max.y), 0.0f);
            float dz = std::max(std::max(min.z - point.z, point.z - max.z), 0.0f);
            return std::sqrt(std::max(0.f,dx * dx + dy * dy + dz * dz));
        }
    };

    // Each voxel is represented by an axis-aligned bounding box
    struct Voxel {
        glm::vec3 center;
        float extent;

        __host__ __device__ inline float bsphere_radius() const
        {
            return extent * sqrt(3.f) * 0.5f;
        }        

        __host__ __device__ inline float inner_bsphere_radius() const
        {
            return extent * 0.5f;
        } 
    };

#ifdef TCNN_NAMESPACE_BEGIN

    struct VoxelGrid{
        // the grids stored as a bitfield
    // tcnn::GPUMemory<uint8_t> grid;
        AABox<glm::vec3> aabb = { glm::vec3(2e32f), glm::vec3(-2e32f)};
        float voxel_size = 0.f;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> grid_handles[NUM_LODS()];
    };
    class NeuralNet {
        public:
            uint32_t n_inference_coords;

            tcnn::json config;

            tcnn::default_rng_t rng{1339};

            //// Optimisation Data
            tcnn::GPUMemory<float> validation_inputs;

            // Auxiliary matrices for training throughput
            tcnn::GPUMatrix<float> training_target_batch;
            tcnn::GPUMatrix<float> training_input_batch;

            // Auxiliary matrices for evaluation
            tcnn::GPUMatrix<float> prediction;
            tcnn::GPUMatrix<float> inference_input_batch;

            // Auxiliary matrices for reference computation
            tcnn::GPUMatrix<float> ref_output_accumulated;

            //// Optimisation Model

            std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> loss;
            std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
            std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> network;
            std::shared_ptr<
                tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>
                trainer;

            //// Optimization temporaries
            float tmp_loss = 0;
            uint32_t tmp_loss_counter = 0;
            uint32_t training_step = 0;
            std::chrono::steady_clock::time_point begin_time;
    };

    struct ThroughputNN : public NeuralNet{
        tcnn::GPUMatrix<float> valid_training_target_batch;
        tcnn::GPUMatrix<float> valid_training_input_batch;
    };
    struct VisibilityNN : public NeuralNet{
        tcnn::GPUMatrix<float> visualisation_input_batch;
    };

#else
    class NeuralNet;
    struct ThroughputNN;
    struct VisibilityNN;
    struct VoxelGrid;
#endif

    std::unique_ptr<ThroughputNN> throughput_nn;
    std::unique_ptr<VisibilityNN> visibility_nn;

    char weights_directory[MAX_FILENAME_SIZE] = "./";
    char weights_filename_prefix[MAX_FILENAME_SIZE] = "";

    std::string get_complete_weights_prefix_path(){
        return std::string("") + weights_directory +  weights_filename_prefix + "_" + std::to_string(VOXEL_GRID_MAX_RES());
    }

    std::string get_optimal_threshold_path(uint32_t lod){
        return std::string("") + weights_directory + "../optimal_thresholds/" +  weights_filename_prefix + "_lod_" + std::to_string(lod) + "_optimal_thresholds.bin";
    }

    ThroughputNN* throughput_neural_net();
    VisibilityNN* visibility_neural_net();

    RenderCudaBinding *backend;

    std::vector<std::string> point_cloud_base_filenames;
    std::string scene_name = "";

    std::unique_ptr<VoxelGrid> voxel_grid;

    // Scene bounding box info
    AABox<glm::vec3> scene_aabb = { glm::vec3(2e32f), glm::vec3(-2e32f)};

    int current_lod = 0;

    // The mapping from sparse voxel linear indexes to morton encoded 3d positions
    uint32_t *morton_sparse_voxel_idx_mapping_gpu = nullptr;
    // A dense array with the cumulated valid samples recorded during inference for all lods
    uint32_t* inferences_sample_count_grid = nullptr;
    uint32_t* inference_total_sample_count = nullptr;

    std::vector<uint32_t> morton_sparse_voxel_idx_mappings[NUM_LODS()];
    int sparse_voxel_count = 0;

    struct Triangle {
        glm::vec3 v1;
        glm::vec3 v2;
        glm::vec3 v3;
        glm::vec3 normal;
        glm::uvec3 voxel_pos_idx;
    };

    int batch_index = 0;
    bool first_init = false;
    bool model_changed = false;
    bool throughput_model_changed = false;
    bool visibility_model_changed = false;
    bool needs_rerender = false;
    bool lod_grids_generated = false;
    bool recompute_inference = false;

    bool scene_loaded = false;

    NeuralLodLearningParams params;

    NeuralLodLearning(RenderCudaBinding *backend);
    ~NeuralLodLearning();

    VoxelGrid* get_voxel_grid() const;

    int get_current_lod() const{
        return current_lod;
    }

    bool throughput_log_learning() const{
        return params.throughput_nn.log_learning;
    }

    void set_lod(int lod) {   
        int new_lod = std::min(std::max(0,lod),int(NUM_LODS()-1));            
        int previous_lod = current_lod;
        current_lod = new_lod;
        set_constants();
        int diff_lod = previous_lod - current_lod;
        if (diff_lod != 0) {
            params.selected_voxel_idx.x =
                diff_lod < 0 ? params.selected_voxel_idx.x >> abs(diff_lod)
                                : params.selected_voxel_idx.x << abs(diff_lod);
            params.selected_voxel_idx.y =
                diff_lod < 0 ? params.selected_voxel_idx.y >> abs(diff_lod)
                                : params.selected_voxel_idx.y << abs(diff_lod);
            params.selected_voxel_idx.z =
                diff_lod < 0 ? params.selected_voxel_idx.z >> abs(diff_lod)
                                : params.selected_voxel_idx.z << abs(diff_lod);
        }
        needs_rerender = true;
    }

    float get_voxel_extent_dilation_outward() const{
        return params.visibility_nn.voxel_extent_dilation_outward;
    }

    bool ui_and_state(bool& renderer_changed) override;
    std::string name() const override;
    void release_resources();

    void initialize(const int fb_width, const int fb_height) override;
    void release_scene_resources();
    void update_scene_from_backend(const Scene &scene) override;
    void compute_voxel_grid_aabb();

    void set_constants();

    void render();

    /************** Optimization *********************/

    void initialize_throughput_nn_data();

    void initialize_throughput_nn_model();

    void initialize_visibility_nn_data();

    void initialize_visibility_nn_model();

    void optimise_thresholds_for_voxel_grid(ThresholdOptimMetrics metric);
    void throughput_network_one_step_optim();
    void visibility_network_one_step_optim();

    /********* Threshold updates ********/

    void update_grid_with_fixed_thresholds();
    void update_grid_with_thresholds(const std::vector<float> &thresholds, uint32_t lod);

    void write_weights_to_disk(
        const std::string &filename,
        NeuralNet* neural_net,
        bool serialize_optimizer = false);

    void write_config_to_disk(
        const std::string &filename,
        NeuralNet* neural_net);

    void load_weights_from_disk(
        const std::string &filename,
        NeuralNet* neural_net);

    // Function to write a vector of floats to a binary file
    void write_vector_to_file(const std::vector<float> &data, const std::string &filename)
    {
        // Open the binary file for writing
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            println(CLL::CRITICAL, "Failed to open file %s", filename.c_str());

        file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));

        if (file.bad())
            println(CLL::CRITICAL, "Failed to write data to file %s", filename.c_str());
        else
            println(
                CLL::INFORMATION, "Successfuly written RoC data to file %s", filename.c_str());

        file.close();
    }

    std::vector<float> read_vector_from_file(const std::string &filename, size_t num_floats)
    {
        // Open the binary file for writing
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            println(CLL::CRITICAL, "Failed to open file %s", filename.c_str());

        std::vector<float> buffer;
        buffer.resize(num_floats);
        file.read(reinterpret_cast<char *>(buffer.data()), num_floats * sizeof(float));

        file.close();

        return buffer;
    }
};

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
inline __host__ __device__ uint32_t Part1By1(uint32_t x) {
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}
inline __host__ __device__ uint32_t Part1By2(uint32_t x) {
  x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}
inline __host__ __device__ uint32_t encode_morton2(uint32_t x, uint32_t y) {
	return (Part1By1(y) << 1) + Part1By1(x);
}
inline __host__ __device__ uint32_t encode_morton3(uint32_t x, uint32_t y, uint32_t z) {
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
inline __host__ __device__ uint32_t Compact1By1(uint32_t x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
inline __host__ __device__ uint32_t Compact1By2(uint32_t x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

inline __host__ __device__ uint32_t DecodeMorton2X(uint32_t code)
{
  return Compact1By1(code >> 0);
}

inline __host__ __device__ uint32_t DecodeMorton2Y(uint32_t code)
{
  return Compact1By1(code >> 1);
}

inline __host__ __device__ uint32_t DecodeMorton3X(uint32_t code)
{
  return Compact1By2(code >> 0);
}

inline __host__ __device__ uint32_t DecodeMorton3Y(uint32_t code)
{
  return Compact1By2(code >> 1);
}

inline __host__ __device__ uint32_t DecodeMorton3Z(uint32_t code)
{
  return Compact1By2(code >> 2);
}

inline __host__ __device__ glm::uvec3 decode_morton3(uint32_t code)
{
  return glm::uvec3(DecodeMorton3X(code),DecodeMorton3Y(code),DecodeMorton3Z(code));
}

NEURAL_LOD_LEARNING_NAMESPACE_END

#endif