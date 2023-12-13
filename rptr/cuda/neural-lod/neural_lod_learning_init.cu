#include <tiny-cuda-nn/common.h>

#include "neural_lod_learning.h"
#include "neural_lod_learning_common.cuh"
#include "neural_lod_learning_grid.cuh"

#include "scene.h"

#include <set>

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN

std::string NeuralLodLearning::name() const
{
    return "Cuda Voxelize Extension";
}

NeuralLodLearning::NeuralLodLearning(RenderCudaBinding *backend) : backend(backend) {}

NeuralLodLearning::~NeuralLodLearning()
{
    release_resources();
}

NeuralLodLearning::ThroughputNN* NeuralLodLearning::throughput_neural_net(){
    return throughput_nn.get();
};

NeuralLodLearning::VisibilityNN* NeuralLodLearning::visibility_neural_net(){
    return visibility_nn.get();
};

NeuralLodLearning::VoxelGrid* NeuralLodLearning::get_voxel_grid() const{
    return voxel_grid.get();
}

void NeuralLodLearning::release_resources()
{
    release_scene_resources();
}

void NeuralLodLearning::release_scene_resources()
{
    println(CLL::INFORMATION, "Releasing scene resources");
    if (morton_sparse_voxel_idx_mapping_gpu) {
        checkCuda(cudaFree(morton_sparse_voxel_idx_mapping_gpu));
        morton_sparse_voxel_idx_mapping_gpu = nullptr;
    }
    if(inferences_sample_count_grid){
        checkCuda(cudaFree(inferences_sample_count_grid));
        inferences_sample_count_grid = nullptr;
    }
    if(inference_total_sample_count){
        checkCuda(cudaFree(inference_total_sample_count));
        inference_total_sample_count = nullptr;
    }
}

void NeuralLodLearning::update_scene_from_backend(const Scene &scene)
{
    println(CLL::INFORMATION, "Updating scene from backend");

    release_resources();

    if(voxel_grid == nullptr)
        voxel_grid = std::make_unique<VoxelGrid>();

    for (auto &&filename : scene.filenames) {
        auto point_cloud_base_filename =
            filename.substr(0, filename.find_last_of(".")).append(".obj");
        point_cloud_base_filenames.emplace_back(point_cloud_base_filename);

        // Todo : Support more than one scene at a time
        if (scene.filenames.size() > 1) {
            throw_error("Currently only one scene file at a time can be processed");
        }
        scene_name = scene.filenames[0].substr(0, scene.filenames[0].find_last_of("."));
        scene_name = scene_name.substr(scene_name.find_last_of('/')+1);

        // Set a default weights directory
        strncpy(weights_directory,
                filename.substr(0, filename.find_last_of("/") + 1).c_str(),
                sizeof(weights_directory) - 1);
    }

    scene_aabb = { glm::vec3(2e32f), glm::vec3(-2e32f)};

    for (auto const &instance : scene.instances) {
        auto &parameterized_mesh = scene.parameterized_meshes[instance.parameterized_mesh_id];
        auto &mesh = scene.meshes[parameterized_mesh.mesh_id];

        const auto &animData = scene.animation_data.at(instance.animation_data_index);
        constexpr uint32_t frame = 0;
        const glm::mat4 instance_transform = animData.dequantize(instance.transform_index, frame);

        println(CLL::VERBOSE,
                "[NeuralLodLearning] Processing mesh with %lu geometries (%lu triangles)",
                mesh.geometries.size(),
                mesh.num_tris());

        if (mesh.num_tris() == 0)
            return;

        bool per_triangle_ids = parameterized_mesh.per_triangle_materials();

        len_t mesh_tri_idx_base = 0;
        for (int i = 0, ie = mesh.num_geometries(); i < ie; ++i) {
            Geometry currentGeom = mesh.geometries[i];
            int material_offset = parameterized_mesh.material_offset(i);

            for (int tri_idx = 0, tri_idx_end = currentGeom.num_tris(); tri_idx < tri_idx_end;
                 ++tri_idx) {
                int material_id = material_offset;
                if (per_triangle_ids)
                    material_id +=
                        parameterized_mesh.triangle_material_id(mesh_tri_idx_base + tri_idx);
                glm::vec3 v0, v1, v2;
                currentGeom.tri_positions(tri_idx, v0, v1, v2);
                v0 = glm::vec3(instance_transform * glm::vec4(v0, 1.0f));
                v1 = glm::vec3(instance_transform * glm::vec4(v1, 1.0f));
                v2 = glm::vec3(instance_transform * glm::vec4(v2, 1.0f));

                scene_aabb.min = min(min(scene_aabb.min, v0), min(v1, v2));
                scene_aabb.max = max(max(scene_aabb.max, v0), max(v1, v2));
            }

            mesh_tri_idx_base += currentGeom.num_tris();
        }
    }

    if (!scene_loaded) {
        scene_loaded = true;
        first_init = true;
    }
}

void NeuralLodLearning::compute_voxel_grid_aabb()
{
    release_resources();

    // This is coherent with the voxelisation done in the cuda-voxelize library
    voxel_grid->aabb = { scene_aabb.min, scene_aabb.max};
    glm::vec3 lengths =
        scene_aabb.max - scene_aabb.min;  // check length of given bbox in every direction
    float max_length = glm::max(lengths.x, glm::max(lengths.y, lengths.z));  // find max length
    for (unsigned int i = 0; i < 3; i++) {  // for every direction (X,Y,Z)
        if (max_length == lengths[i]) {
            continue;
        } else {
            float delta = max_length - lengths[i];  // compute difference between largest
                                                    // length and current (X,Y or Z) length
            voxel_grid->aabb.min[i] =
                scene_aabb.min[i] -
                (delta / 2.0f);  // pad with half the difference before current min
            voxel_grid->aabb.max[i] =
                scene_aabb.max[i] +
                (delta / 2.0f);  // pad with half the difference behind current max
        }
    }

// Potential regularisation if used aswell in the cuda-voxelizer
#ifdef VOXELIZER_REGULARISATION
    glm::vec3 epsilon = (voxel_grid->aabb.max - voxel_grid->aabb.min) * 1e-4f;
    voxel_grid->aabb.min -= epsilon;
    voxel_grid->aabb.max += epsilon;
#endif

    glm::vec3 unit = glm::vec3((voxel_grid->aabb.max.x - voxel_grid->aabb.min.x) / VOXEL_GRID_MAX_RES(),
                               (voxel_grid->aabb.max.y - voxel_grid->aabb.min.y) / VOXEL_GRID_MAX_RES(),
                               (voxel_grid->aabb.max.z - voxel_grid->aabb.min.z) / VOXEL_GRID_MAX_RES());
    assert(abs(unit.x - unit.y) < 1e5 && abs(unit.y - unit.z) < 1e5);
    voxel_grid->voxel_size = unit.x;

    println(CLL::INFORMATION,
            "Scene Bounding Box: (%g,%g,%g)-(%g,%g,%g)",
            scene_aabb.min.x,
            scene_aabb.min.y,
            scene_aabb.min.z,
            scene_aabb.max.x,
            scene_aabb.max.y,
            scene_aabb.max.z);

    println(CLL::INFORMATION,
            "Voxels Bounding Box: (%f,%f,%f)-(%f,%f,%f)",
            voxel_grid->aabb.min.x,
            voxel_grid->aabb.min.y,
            voxel_grid->aabb.min.z,
            voxel_grid->aabb.max.x,
            voxel_grid->aabb.max.y,
            voxel_grid->aabb.max.z);
    println(CLL::INFORMATION, "Grid max res: %i", VOXEL_GRID_MAX_RES());
    println(CLL::INFORMATION, "voxel_size: %f", voxel_grid->voxel_size);

    nanovdb::GridBuilder<float> lod_builders[NUM_LODS()];
    for (size_t lod_level = 0; lod_level < NUM_LODS(); lod_level++) {
        lod_builders[lod_level] =
            nanovdb::GridBuilder<float>(0.0f, nanovdb::GridClass::VoxelVolume);
        morton_sparse_voxel_idx_mappings[lod_level].clear();
    }

    std::set<uint32_t> sparse_sets[NUM_LODS()];
    for (auto point_cloud_base_filename : point_cloud_base_filenames) {
        std::string point_cloud_max_res_filename = point_cloud_base_filename;
#ifdef VOXELIZER_REGULARISATION
        point_cloud_max_res_filename += "_eps_padded";
#endif
        point_cloud_max_res_filename += std::string("_") + std::to_string(VOXEL_GRID_MAX_RES()) + "_pointcloud.obj";

        if (!std::ifstream(point_cloud_max_res_filename.c_str()).good()) {
            throw_error("Point cloud file not found");
        } else {
            println(CLL::INFORMATION,
                    "Point cloud file : %s found",
                    point_cloud_max_res_filename.c_str());
            std::string line;
            std::ifstream file;
            file.open(point_cloud_max_res_filename);

            if (!file.is_open()) {
                throw_error("Point cloud file could not be oppened");
            }

            while (std::getline(file, line)) {
                // one vertex per line
                std::istringstream iss(line);
                std::string item;
                while (std::getline(iss, item, ' ')) {
                    if (item == "v") {
                        // convert floating point values to indexes
                        std::string pos[3];
                        int index_pos[3];
                        for (size_t i = 0; i < 3; i++) {
                            std::getline(iss, pos[i], ' ');
                            index_pos[i] = int(std::stof(pos[i]));
                        }
                        // std::cout << "vertex : " << index_pos[0] << " " << index_pos[1] << "
                        // " << index_pos[2] << std::endl;
                        for (size_t lod_level = 0; lod_level < NUM_LODS(); lod_level++) {
                            uint32_t x_lod = index_pos[0] >> lod_level;
                            uint32_t y_lod = index_pos[1] >> lod_level;
                            uint32_t z_lod = index_pos[2] >> lod_level;
                            const float initial_threshold = 0.5f;
                            lod_builders[lod_level].getAccessor().setValue(
                                nanovdb::Coord(x_lod, y_lod, z_lod), initial_threshold);
                            int current_lod_res = VOXEL_GRID_MAX_RES() >> lod_level;
                            uint32_t i = coord_3d_to_1d(x_lod,y_lod,z_lod,current_lod_res,current_lod_res,current_lod_res);
                            sparse_sets[lod_level].insert(encode_morton3(x_lod,y_lod,z_lod));
                        }
                    } else {
                        println(CLL::WARNING,
                                "Encountered unsupported specifier in point cloud file! "
                                "Skipping to next vertex");
                    }
                }
                // std::cout << line << std::endl;
            }
        }
    }

    // Finalize grids
    float step_size = voxel_grid->voxel_size;
    float total_mb_estimate = 0.f;
    for (size_t lod_level = 0; lod_level < NUM_LODS(); lod_level++) {
        voxel_grid->grid_handles[lod_level] =
            lod_builders[lod_level].getHandle<nanovdb::AbsDiff, nanovdb::CudaDeviceBuffer>(
                step_size, nanovdb::Vec3d(voxel_grid->aabb.min.x, voxel_grid->aabb.min.y, voxel_grid->aabb.min.z));
        voxel_grid->grid_handles[lod_level].deviceUpload();

        step_size *= 2;

        // Some stats
        uint32_t lod_sparse_voxel_count = voxel_grid->grid_handles[lod_level].gridMetaData()->activeVoxelCount();
        uint32_t max_voxel_count =  1 << ((VOXEL_GRID_MAX_POWER_2()-lod_level) * 3);
        float sparsity = 100.f - (100.f * lod_sparse_voxel_count / float(max_voxel_count));
        float mb_size = float(lod_sparse_voxel_count * 4) * 1e-6;
        total_mb_estimate += mb_size;
        println(CLL::INFORMATION,
                "LOD %u : %u active voxels, sparsity : %.3f %, ~ %.4f MB",
                lod_level,
                voxel_grid->grid_handles[lod_level].gridMetaData()->activeVoxelCount(),
                sparsity,
                mb_size);
    }
    println(CLL::INFORMATION, "Total approximate size of grids: ~ %.4f MB", total_mb_estimate);

    sparse_voxel_count = voxel_grid->grid_handles[0].gridMetaData()->activeVoxelCount();

    for (size_t lod = 0; lod < NUM_LODS(); lod++) {
        morton_sparse_voxel_idx_mappings[lod] =
            std::vector<uint32_t>(sparse_sets[lod].begin(), sparse_sets[lod].end());

        assert(morton_sparse_voxel_idx_mappings[lod].size() ==
               voxel_grid->grid_handles[lod].gridMetaData()->activeVoxelCount());
        std::sort(morton_sparse_voxel_idx_mappings[lod].begin(),morton_sparse_voxel_idx_mappings[lod].end());
    }

    // Copy largest res to gpu side
    if (morton_sparse_voxel_idx_mapping_gpu) {
        cudaFree(morton_sparse_voxel_idx_mapping_gpu);
    }
    checkCuda(cudaMalloc(&morton_sparse_voxel_idx_mapping_gpu, sizeof(uint32_t) * sparse_voxel_count));
    checkCuda(cudaMemcpy(morton_sparse_voxel_idx_mapping_gpu,
                         morton_sparse_voxel_idx_mappings[0].data(),
                         sizeof(uint32_t) * sparse_voxel_count,
                         cudaMemcpyHostToDevice));

    lod_grids_generated = true;

    printf("voxels treeIndexBbox: [%u,%u,%u] -> [%u,%u,%u]\n",
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[0],
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[1],
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[2],
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[0],
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[1],
           voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[2]);
    println(CLL::INFORMATION,
            "voxels occupying worldBBox : [%f %f %f] -> [%f %f %f]",
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][0],
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][1],
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][2],
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][0],
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][1],
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][2]);
    println(CLL::INFORMATION,
            "voxels worldBBox : [%f %f %f] -> [%f %f %f] (should equal the voxel_grid->aabb "
            "computed above)",
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][0] -
                voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[0] * voxel_grid->voxel_size,
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][1] -
                voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[1] * voxel_grid->voxel_size,
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[0][2] -
                voxel_grid->grid_handles[0].grid<float>()->indexBBox().min()[2] * voxel_grid->voxel_size,
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][0] +
                ((VOXEL_GRID_MAX_RES()-1) - voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[0]) * voxel_grid->voxel_size,
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][1] +
                ((VOXEL_GRID_MAX_RES()-1) - voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[1]) * voxel_grid->voxel_size,
            voxel_grid->grid_handles[0].grid<float>()->worldBBox()[1][2] +
                ((VOXEL_GRID_MAX_RES()-1) - voxel_grid->grid_handles[0].grid<float>()->indexBBox().max()[2]) * voxel_grid->voxel_size);

    println(CLL::INFORMATION,
            "scene bbox : [%f %f %f] -> [%f %f %f] found",
            scene_aabb.min[0],
            scene_aabb.min[1],
            scene_aabb.min[2],
            scene_aabb.max[0],
            scene_aabb.max[1],
            scene_aabb.max[2]);

    assert(voxel_grid->aabb.min.x <= scene_aabb.min.x);
    assert(voxel_grid->aabb.min.y <= scene_aabb.min.y);
    assert(voxel_grid->aabb.min.z <= scene_aabb.min.z);
    assert(voxel_grid->aabb.max.x >= scene_aabb.max.x);
    assert(voxel_grid->aabb.max.y >= scene_aabb.max.y);
    assert(voxel_grid->aabb.max.z >= scene_aabb.max.z);

    println(CLL::INFORMATION,
            "voxelSize : %f %f %f\n",
            voxel_grid->grid_handles[0].grid<float>()->voxelSize()[0],
            voxel_grid->grid_handles[0].grid<float>()->voxelSize()[1],
            voxel_grid->grid_handles[0].grid<float>()->voxelSize()[2]);
}

void NeuralLodLearning::initialize(const int /*fb_width*/, const int /*fb_height*/)
{
    if (!scene_loaded)
        return;

    if(first_init){
        println(CLL::INFORMATION, "Initializing scene");

        if(voxel_grid == nullptr)
            voxel_grid = std::make_unique<VoxelGrid>();

        compute_voxel_grid_aabb();
        set_constants();            
        
        // Load optimal thresholds if available
        for (size_t lod = 0; lod < NUM_LODS(); lod++)
        {
            auto threshold_filename = get_optimal_threshold_path(lod);
            std::ifstream infile(threshold_filename);
            if(infile.good()){
                uint32_t lod_sparse_voxel_count = voxel_grid->grid_handles[lod].gridMetaData()->activeVoxelCount();
                auto thresholds = read_vector_from_file(threshold_filename,lod_sparse_voxel_count);
                update_grid_with_thresholds(thresholds,lod);
                println(CLL::INFORMATION,"Loaded optimal thresholds from disk for lod %i",lod);
            }else{
                println(CLL::WARNING,"No optimal thresholds found on disk at : %s for lod %i",threshold_filename.c_str(),lod);
            }
        }
    }

    if(first_init || throughput_model_changed || model_changed){
        throughput_model_changed = false;
        println(CLL::INFORMATION, "Initializing Throughput NN");
        if(throughput_nn == nullptr)
            throughput_nn = std::make_unique<ThroughputNN>();

        initialize_throughput_nn_data();

        initialize_throughput_nn_model();

        throughput_nn->tmp_loss = 0;
        throughput_nn->tmp_loss_counter = 0;

        throughput_nn->begin_time = std::chrono::steady_clock::now();

        throughput_nn->training_step = 0;

        if(first_init){
            auto weights_filename = get_complete_weights_prefix_path() + "_throughput_weights.json";
            std::ifstream infile(weights_filename);
            if(infile.good()){
                load_weights_from_disk(
                    get_complete_weights_prefix_path() + "_throughput_weights.json",
                    throughput_nn.get());
            }else{
                println(CLL::WARNING,"No precomputed weights for throughput found on disk at : %s",weights_filename.c_str());
            }

            
            
        }
    }

    if(first_init || visibility_model_changed || model_changed){
        visibility_model_changed = false;
        println(CLL::INFORMATION, "Initializing Visibility NN");
        if(visibility_nn == nullptr)
            visibility_nn = std::make_unique<VisibilityNN>();

        initialize_visibility_nn_data();

        initialize_visibility_nn_model();

        visibility_nn->tmp_loss = 0;
        visibility_nn->tmp_loss_counter = 0;

        visibility_nn->begin_time = std::chrono::steady_clock::now();

        visibility_nn->training_step = 0;

        if(first_init){
            auto weights_filename = get_complete_weights_prefix_path() + "_visibility_weights.json";
            std::ifstream infile(weights_filename);
            if(infile.good()){
                load_weights_from_disk(
                    get_complete_weights_prefix_path() + "_visibility_weights.json",
                    visibility_nn.get());
            }else{
                println(CLL::WARNING,"No precomputed weights for visibility found on disk at : %s",weights_filename.c_str());
            }
        }
    }

    if(model_changed){
        model_changed = false;
    }
    first_init = false;

    batch_index = 0;
}

void NeuralLodLearning::initialize_throughput_nn_data()
{
    throughput_nn->n_inference_coords = (params.throughput_nn.inference_2d_sampling_res.x * params.throughput_nn.inference_2d_sampling_res.y) *
                             (params.throughput_nn.inference_2d_sampling_res.x * params.throughput_nn.inference_2d_sampling_res.y);

    assert(params.throughput_nn.batch_size <= DEFAULT_RAY_QUERY_BUDGET * 2);

    throughput_nn->validation_inputs = tcnn::GPUMemory<float>(throughput_nn->n_inference_coords * throughput_n_input_dims);
    std::vector<float> host_validation_inputs = std::vector<float>(throughput_nn->n_inference_coords * throughput_n_input_dims);

    Voxel voxel = compute_lod_voxel(
        params.selected_voxel_idx, voxel_grid->aabb, voxel_grid->voxel_size, current_lod);

    glm::vec3 normalized_center = (voxel.center - voxel_grid->aabb.min) / voxel_grid->aabb.extent();

    int idx = 0;
    for (int phi_i_idx = 0; phi_i_idx < params.throughput_nn.inference_2d_sampling_res.y; ++phi_i_idx) {
        for (int theta_i_idx = 0; theta_i_idx < params.throughput_nn.inference_2d_sampling_res.x; ++theta_i_idx) {
            for (int phi_o_idx = 0; phi_o_idx < params.throughput_nn.inference_2d_sampling_res.y; ++phi_o_idx) {
                for (int theta_o_idx = 0; theta_o_idx < params.throughput_nn.inference_2d_sampling_res.x;
                     ++theta_o_idx) {
                    float theta_o =
                        (float)(theta_o_idx + 0.5) / (float)params.throughput_nn.inference_2d_sampling_res.x * M_PI;
                    float phi_o =
                        (float)(phi_o_idx + 0.5) / (float)params.throughput_nn.inference_2d_sampling_res.y * 2.f * M_PI -
                        M_PI;
                    float theta_i =
                        (float)(theta_i_idx + 0.5) / (float)params.throughput_nn.inference_2d_sampling_res.x * M_PI;
                    float phi_i =
                        (float)(phi_i_idx + 0.5) / (float)params.throughput_nn.inference_2d_sampling_res.y * 2.f * M_PI -
                        M_PI;

                    glm::vec3 wo = spherical_to_cartesian(theta_o,phi_o);
                    glm::vec3 wi = spherical_to_cartesian(theta_i,phi_i);

                    host_validation_inputs[idx + 0] = normalized_center.x;
                    host_validation_inputs[idx + 1] = normalized_center.y;
                    host_validation_inputs[idx + 2] = normalized_center.z;

                    glm::vec3 input_wo = (wo + 1.0f) * 0.5f;
                    host_validation_inputs[idx + 3] = input_wo.x;
                    host_validation_inputs[idx + 4] = input_wo.y;
                    host_validation_inputs[idx + 5] = input_wo.z;

                    glm::vec3 input_wi = (wi + 1.0f) * 0.5f;
                    host_validation_inputs[idx + 6] = input_wi.x;
                    host_validation_inputs[idx + 7] = input_wi.y;
                    host_validation_inputs[idx + 8] = input_wi.z;

                    // Make sure the inverse function is correct
                    assert(compute_inference_idx_for_throughput(
                               wo,
                               wi,
                               params.throughput_nn.inference_2d_sampling_res,
                               throughput_n_input_dims) == idx);

                    int test_index =
                        coord_4d_to_1d(theta_o_idx,
                                       phi_o_idx,
                                       theta_i_idx,
                                       phi_i_idx,
                                       params.throughput_nn.inference_2d_sampling_res.x,
                                       params.throughput_nn.inference_2d_sampling_res.y,
                                       params.throughput_nn.inference_2d_sampling_res.x,
                                       params.throughput_nn.inference_2d_sampling_res.y);

                    assert(idx == test_index * throughput_n_input_dims);

                    idx += throughput_n_input_dims;
                }
            }
        }
    }

    throughput_nn->validation_inputs.copy_from_host(host_validation_inputs.data());

    // Auxiliary matrices for training
    reinitialize_object(throughput_nn->training_input_batch,
                        tcnn::GPUMatrix<float>(throughput_n_input_dims, params.throughput_nn.batch_size));
    throughput_nn->training_input_batch.initialize_constant(0.f);

    reinitialize_object(throughput_nn->training_target_batch,
                        tcnn::GPUMatrix<float>(throughput_n_output_dims, params.throughput_nn.batch_size));
    throughput_nn->training_target_batch.initialize_constant(0.f);
    
    // Auxiliary matrices for training
    reinitialize_object(throughput_nn->valid_training_input_batch,
                        tcnn::GPUMatrix<float>(throughput_n_input_dims, params.throughput_nn.batch_size));
    throughput_nn->valid_training_input_batch.initialize_constant(0.f);

    reinitialize_object(throughput_nn->valid_training_target_batch,
                        tcnn::GPUMatrix<float>(throughput_n_output_dims, params.throughput_nn.batch_size));
    throughput_nn->valid_training_target_batch.initialize_constant(0.f);

    // Auxiliary matrices for predictions
    reinitialize_object(throughput_nn->inference_input_batch,
                        tcnn::GPUMatrix<float>(throughput_nn->validation_inputs.data(),
                                               throughput_n_input_dims,
                                               throughput_nn->n_inference_coords));
    // don't override predictions

    reinitialize_object(throughput_nn->prediction,
                        tcnn::GPUMatrix<float>(throughput_n_output_dims, throughput_nn->n_inference_coords));
    throughput_nn->prediction.initialize_constant(0.f);

    // Auxiliary matrices for reference
    // Last additional dimension used to count the number of samples
    reinitialize_object(throughput_nn->ref_output_accumulated,
                        tcnn::GPUMatrix<float>((throughput_n_output_dims + 1), throughput_nn->n_inference_coords));
    // Make sure to initialize the matrix with 0s
    throughput_nn->ref_output_accumulated.initialize_constant(0.f);
}

void NeuralLodLearning::initialize_throughput_nn_model()
{
    throughput_nn->config = {
        {"loss", {{"otype", loss_types[params.throughput_nn.loss]}}},
        {"optimizer", 
            {{"otype", "Ema"},
                {"decay", 0.95},
                {"nested", {
                    {"otype", "ExponentialDecay"},
                    {"decay_start", params.throughput_nn.optimizer_options.decay_start},
                    {"decay_interval", params.throughput_nn.optimizer_options.decay_interval},
                    {"decay_base", params.throughput_nn.optimizer_options.decay_base},
                    {"nested", {
                        {"otype", optimizer_types[params.throughput_nn.optimizer]},
                        {"learning_rate", params.throughput_nn.optimizer_options.learning_rate},
                        {"beta1",params.throughput_nn.optimizer_options.beta1},
                        {"beta2", params.throughput_nn.optimizer_options.beta2},
                        {"epsilon", params.throughput_nn.optimizer_options.epsilon},
                        {"l2_reg", params.throughput_nn.optimizer_options.l2_reg},
                        {"relative_decay", params.throughput_nn.optimizer_options.relative_decay},
                        {"absolute_decay", params.throughput_nn.optimizer_options.absolute_decay},
                        {"adabound", params.throughput_nn.optimizer_options.adabound},
                    }}
                }}
            }
        },
        {"encoding",
         {{"otype", "Composite"},
          {"nested",
           {{
                {"n_dims_to_encode", 3},  // Voxel position
                {"otype", encoding_types[params.throughput_nn.voxel_encoding]},
                // hashgrid/grid encodings
                {"n_levels", params.throughput_nn.voxel_grid_options.n_levels},
                {"n_features_per_level", params.throughput_nn.voxel_grid_options.n_features_per_level},
                {"log2_hashmap_size", params.throughput_nn.voxel_grid_options.log2_hashmap_size},
                {"base_resolution", params.throughput_nn.voxel_grid_options.base_resolution},
                {"per_level_scale", params.throughput_nn.voxel_grid_options.per_level_scale},
                {"interpolation",  interpolation_types[params.throughput_nn.voxel_grid_options.interpolation]}
            },
            {
                {"n_dims_to_encode", 3},  // wo
                {"otype", encoding_types[params.throughput_nn.outgoing_encoding]},
                // hashgrid/grid encodings
                {"n_levels", params.throughput_nn.outgoing_grid_options.n_levels},
                {"n_features_per_level", params.throughput_nn.outgoing_grid_options.n_features_per_level},
                {"log2_hashmap_size", params.throughput_nn.outgoing_grid_options.log2_hashmap_size},
                {"base_resolution", params.throughput_nn.outgoing_grid_options.base_resolution},
                {"per_level_scale", params.throughput_nn.outgoing_grid_options.per_level_scale},
                {"interpolation",  interpolation_types[params.throughput_nn.outgoing_grid_options.interpolation]},
                // sh encoding
                {"degree", params.throughput_nn.outgoing_sh_options.max_degree}
            },
            {
                {"n_dims_to_encode", 3},  // wi
                {"otype", encoding_types[params.throughput_nn.incident_encoding]},
                // hashgrid/grid encodings
                {"n_levels", params.throughput_nn.incident_grid_options.n_levels},
                {"n_features_per_level", params.throughput_nn.incident_grid_options.n_features_per_level},
                {"log2_hashmap_size", params.throughput_nn.incident_grid_options.log2_hashmap_size},
                {"base_resolution", params.throughput_nn.incident_grid_options.base_resolution},
                {"per_level_scale", params.throughput_nn.incident_grid_options.per_level_scale},
                {"interpolation",  interpolation_types[params.throughput_nn.incident_grid_options.interpolation]},
                // sh encoding
                {"degree", params.throughput_nn.incident_sh_options.max_degree}
            }
            }}}},
        {"network",
         {
             {"otype", network_types[params.throughput_nn.network]},
             {"n_neurons", params.throughput_nn.n_neurons},
             {"n_hidden_layers", params.throughput_nn.n_hidden_layers},
             {"activation", activation_types[params.throughput_nn.activation]},
             {"output_activation", activation_types[params.throughput_nn.output_activation]},
         }},
    };

    tcnn::json encoding_opts = throughput_nn->config.value("encoding", tcnn::json::object());
    tcnn::json loss_opts = throughput_nn->config.value("loss", tcnn::json::object());
    tcnn::json optimizer_opts = throughput_nn->config.value("optimizer", tcnn::json::object());
    tcnn::json network_opts = throughput_nn->config.value("network", tcnn::json::object());

    throughput_nn->loss = std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>>{
        tcnn::create_loss<tcnn::network_precision_t>(loss_opts)};
    throughput_nn->optimizer = std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>>{
        tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_opts)};
    throughput_nn->network = std::make_shared<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>>(
        throughput_n_input_dims, throughput_n_output_dims, encoding_opts, network_opts);

    throughput_nn->trainer = std::make_shared<
        tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(
        throughput_nn->network, throughput_nn->optimizer, throughput_nn->loss);

    println(CLL::INFORMATION,
            "Throughput Network info : \n"
            "\tn_params: %u"
            "\tsize in MB : %f",
            throughput_nn->network->n_params(),
            2.0 * throughput_nn->network->n_params() / 1000000.0);
}

void NeuralLodLearning::initialize_visibility_nn_data()
{
    visibility_nn->n_inference_coords = (params.visibility_nn.inference_3d_sampling_res.x *
                                         params.visibility_nn.inference_3d_sampling_res.y) *
                                        params.visibility_nn.inference_3d_sampling_res.z *
                                        (params.visibility_nn.inference_3d_sampling_res.x *
                                         params.visibility_nn.inference_3d_sampling_res.y) *
                                        params.visibility_nn.inference_3d_sampling_res.z;

    assert(params.visibility_nn.batch_size <= DEFAULT_RAY_QUERY_BUDGET * 2);

    visibility_nn->validation_inputs = tcnn::GPUMemory<float>(visibility_nn->n_inference_coords * visibility_n_input_dims);
    std::vector<float> host_validation_inputs = std::vector<float>(visibility_nn->n_inference_coords * visibility_n_input_dims);

    Voxel voxel = compute_lod_voxel(
        params.selected_voxel_idx, voxel_grid->aabb, voxel_grid->voxel_size, current_lod);
    glm::vec3 normalized_center = (voxel.center - voxel_grid->aabb.min) / voxel_grid->aabb.extent();

    int idx = 0;
    for (int face_idx_po = 0; face_idx_po < params.visibility_nn.inference_3d_sampling_res.z;
         ++face_idx_po) {
        for (int po_idx_v = 0; po_idx_v < params.visibility_nn.inference_3d_sampling_res.y;
             ++po_idx_v) {
            for (int po_idx_u = 0;
                 po_idx_u < params.visibility_nn.inference_3d_sampling_res.x;
                 ++po_idx_u) {
                for (int face_idx_pi = 0;
                     face_idx_pi < params.visibility_nn.inference_3d_sampling_res.z;
                     ++face_idx_pi) {
                    for (int pi_idx_v = 0;
                         pi_idx_v < params.visibility_nn.inference_3d_sampling_res.y;
                         ++pi_idx_v) {
                        for (int pi_idx_u = 0;
                             pi_idx_u < params.visibility_nn.inference_3d_sampling_res.x;
                             ++pi_idx_u) {
                            float pi_u =
                                (float)(pi_idx_u + 0.5) /
                                (float)params.visibility_nn.inference_3d_sampling_res.x;
                            float pi_v = (float)(pi_idx_v + 0.5) /
                                         params.visibility_nn.inference_3d_sampling_res.y;

                            glm::vec3 local_pi = local_voxel_pos_from_uv_and_face(
                                glm::vec2(pi_u, pi_v), (NeuralLodLearning::CubeFace)face_idx_pi);
                            
                            float po_u =
                                (float)(po_idx_u + 0.5) /
                                (float)params.visibility_nn.inference_3d_sampling_res.x;
                            float po_v = (float)(po_idx_v + 0.5) /
                                         params.visibility_nn.inference_3d_sampling_res.y;

                            glm::vec3 local_po = local_voxel_pos_from_uv_and_face(
                                glm::vec2(po_u, po_v), (NeuralLodLearning::CubeFace)face_idx_po);

                            glm::vec3 pi =
                                voxel.center +
                                local_pi *
                                    (1.f + params.visibility_nn.voxel_extent_min_dilation);
                            glm::vec3 po =
                                voxel.center +
                                local_po *
                                    (1.f + params.visibility_nn.voxel_extent_min_dilation);

                            glm::vec3 input_pi = (pi - voxel_grid->aabb.min) / voxel_grid->aabb.extent();
                            glm::vec3 input_po = (po - voxel_grid->aabb.min) / voxel_grid->aabb.extent();

                            host_validation_inputs[idx + 0] = input_pi.x;
                            host_validation_inputs[idx + 1] = input_pi.y;
                            host_validation_inputs[idx + 2] = input_pi.z;

                            host_validation_inputs[idx + 3] = input_po.x;
                            host_validation_inputs[idx + 4] = input_po.y;
                            host_validation_inputs[idx + 5] = input_po.z;

                            int test_index = coord_6d_to_1d(
                                pi_idx_u,
                                pi_idx_v,
                                face_idx_pi,
                                po_idx_u,
                                po_idx_v,
                                face_idx_po,
                                params.visibility_nn.inference_3d_sampling_res.x,
                                params.visibility_nn.inference_3d_sampling_res.y,
                                params.visibility_nn.inference_3d_sampling_res.z,
                                params.visibility_nn.inference_3d_sampling_res.x,
                                params.visibility_nn.inference_3d_sampling_res.y,
                                params.visibility_nn.inference_3d_sampling_res.z);
                            
                            // Make sure the inverse function is correct
                            assert(compute_inference_idx_for_visibility(
                                       local_pi,
                                       local_po,
                                       params.visibility_nn.inference_3d_sampling_res,
                                       visibility_n_input_dims) == idx);

                            assert(idx == test_index * visibility_n_input_dims);

                            idx += visibility_n_input_dims;
                        }
                    }
                }
            }
        }
    }

    visibility_nn->validation_inputs.copy_from_host(host_validation_inputs.data());

    // Auxiliary matrices for training
    reinitialize_object(
        visibility_nn->training_input_batch,
        tcnn::GPUMatrix<float>(visibility_n_input_dims, params.visibility_nn.batch_size));
    visibility_nn->training_input_batch.initialize_constant(0.f);

    reinitialize_object(
        visibility_nn->training_target_batch,
        tcnn::GPUMatrix<float>(visibility_n_output_dims, params.visibility_nn.batch_size));
    visibility_nn->training_target_batch.initialize_constant(0.f);

    // Auxiliary matrices for predictions
    reinitialize_object(visibility_nn->inference_input_batch,
                        tcnn::GPUMatrix<float>(visibility_nn->validation_inputs.data(),
                                               visibility_n_input_dims,
                                               visibility_nn->n_inference_coords));
    // don't override predictions

    reinitialize_object(
        visibility_nn->prediction,
        tcnn::GPUMatrix<float>(visibility_n_output_dims, visibility_nn->n_inference_coords));
    visibility_nn->prediction.initialize_constant(0.f);

    // Auxiliary matrices for reference
    // Last additional dimension used to count the number of samples
    reinitialize_object(visibility_nn->ref_output_accumulated,
                        tcnn::GPUMatrix<float>((visibility_n_output_dims + 1),
                                               visibility_nn->n_inference_coords));
    // Make sure to initialize the matrix with 0s
    visibility_nn->ref_output_accumulated.initialize_constant(0.f);

    // Auxiliary matrix for visualisation only
    reinitialize_object(
        visibility_nn->visualisation_input_batch,
        tcnn::GPUMatrix<float>(visibility_n_input_dims, visibility_nn->n_inference_coords));
    visibility_nn->visualisation_input_batch.initialize_constant(0.f);
}

void NeuralLodLearning::initialize_visibility_nn_model()
{
    visibility_nn->config = {
        {"loss", {{"otype", loss_types[params.visibility_nn.loss]}}},
        {"optimizer",
         {{"otype", "ExponentialDecay"},
          {"decay_start", params.visibility_nn.optimizer_options.decay_start},
          {"decay_interval", params.visibility_nn.optimizer_options.decay_interval},
          {"decay_base", params.visibility_nn.optimizer_options.decay_base},
          {"nested",
           {
               {"otype", optimizer_types[params.visibility_nn.optimizer]},
               {"learning_rate", params.visibility_nn.optimizer_options.learning_rate},
               {"beta1",params.visibility_nn.optimizer_options.beta1},
               {"beta2", params.visibility_nn.optimizer_options.beta2},
               {"epsilon", params.visibility_nn.optimizer_options.epsilon},
               {"l2_reg", params.visibility_nn.optimizer_options.l2_reg},
               {"relative_decay", params.visibility_nn.optimizer_options.relative_decay},
               {"absolute_decay", params.visibility_nn.optimizer_options.absolute_decay},
               {"adabound", params.visibility_nn.optimizer_options.adabound},
               // The following parameters are only used when the optimizer is "Shampoo".
               {"beta3", 0.9f},
               {"beta_shampoo", 0.0f},
               {"identity", 0.0001f},
               {"cg_on_momentum", false},
               {"frobenius_normalization", true},
           }}}},
        {"encoding",
         {
            {"otype", "Composite"},
            {"n_dims_to_encode", visibility_n_input_dims},
            {"nested",
                {
                    {
                        {"otype", "RepeatedComposite"},
                        {"n_repetitions", 2},
                        {"n_dims_to_encode", 6},
                        {"nested",
                            {
                                {
                                    {"n_dims_to_encode", 3},  // pi / po
                                    {"otype", encoding_types[params.visibility_nn.voxel_entry_exit_encoding]},
                                    // hashgrid/grid encodings
                                    {"n_levels", params.visibility_nn.voxel_entry_exit_grid_options.n_levels},
                                    {"n_features_per_level", params.visibility_nn.voxel_entry_exit_grid_options.n_features_per_level},
                                    {"log2_hashmap_size", params.visibility_nn.voxel_entry_exit_grid_options.log2_hashmap_size},
                                    {"base_resolution", params.visibility_nn.voxel_entry_exit_grid_options.base_resolution},
                                    {"per_level_scale", params.visibility_nn.voxel_entry_exit_grid_options.per_level_scale},
                                    {"stochastic_interpolation", params.visibility_nn.position_grid_options.stochastic_interpolation},
                                    {"interpolation",  interpolation_types[params.visibility_nn.voxel_entry_exit_grid_options.interpolation]}
                                },
                            }
                        }
                    },
                }
            },
        }},
        {"network",
         {
             {"otype", network_types[params.visibility_nn.network]},
             {"n_neurons", params.visibility_nn.n_neurons},
             {"n_hidden_layers", params.visibility_nn.n_hidden_layers},
             {"activation", activation_types[params.visibility_nn.activation]},
             {"output_activation", activation_types[params.visibility_nn.output_activation]},
         }},
    };

    tcnn::json encoding_opts = visibility_nn->config.value("encoding", tcnn::json::object());
    tcnn::json loss_opts = visibility_nn->config.value("loss", tcnn::json::object());
    tcnn::json optimizer_opts = visibility_nn->config.value("optimizer", tcnn::json::object());
    tcnn::json network_opts = visibility_nn->config.value("network", tcnn::json::object());

    visibility_nn->loss = std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>>{
        tcnn::create_loss<tcnn::network_precision_t>(loss_opts)};
    visibility_nn->optimizer = std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>>{
        tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_opts)};
    visibility_nn->network = std::make_shared<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>>(
        visibility_n_input_dims, visibility_n_output_dims, encoding_opts, network_opts);

    visibility_nn->trainer = std::make_shared<
        tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(
        visibility_nn->network, visibility_nn->optimizer, visibility_nn->loss);

    println(CLL::INFORMATION,
            "Visibility Network info : \n"
            "\tn_params: %u"
            "\tsize in MB : %f",
            visibility_nn->network->n_params(),
            2.0 * visibility_nn->network->n_params() / 1000000.0);
}

NEURAL_LOD_LEARNING_NAMESPACE_END