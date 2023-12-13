#include <tiny-cuda-nn/common.h>

#include "neural_lod_learning.h"
#include "neural_lod_learning_kernels.cuh"
#include "neural_lod_learning_common.cuh"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN

void NeuralLodLearning::set_constants(){
    checkCuda(cudaMemcpyToSymbol(voxel_grid_aabb_const, &voxel_grid->aabb, sizeof(voxel_grid->aabb)));
    checkCuda(cudaMemcpyToSymbol(voxel_size_const, &voxel_grid->voxel_size, sizeof(voxel_grid->voxel_size)));
    checkCuda(cudaMemcpyToSymbol(current_lod_const, &current_lod, sizeof(current_lod)));
}

void NeuralLodLearning::update_grid_with_fixed_thresholds(){

    for (size_t lod = 0; lod < NUM_LODS(); lod++)
    {
        // Finally write the optimised thresholds to each voxel
        auto grid_d = voxel_grid->grid_handles[lod].deviceGrid<float>();
        uint32_t leaf_count = voxel_grid->grid_handles[lod].gridMetaData()->nodeCount(0);

        float fixed_threshold = params.threshold_fixed_value;

        auto kernel = [grid_d, fixed_threshold] __device__ (const uint64_t n) {
            assert(grid_d->isSequential<0>() == true);
            auto *leaf_d = grid_d->tree().getFirstNode<0>() +
                            (n >> 9);  // this only works if grid->isSequential<0>() == true
            const int i = n & 511;
            if (leaf_d->isActive(i)) {
                const float new_value = fixed_threshold;
                leaf_d->setValueOnly(i, new_value);  // only possible execution divergence
            }
        };

        thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
        thrust::for_each(iter, iter + 512 * leaf_count, kernel);
    }
}


void NeuralLodLearning::update_grid_with_thresholds(const std::vector<float> &cpu_thresholds, uint32_t lod){

    // Finally write the optimised thresholds to each voxel
    auto grid_d = voxel_grid->grid_handles[lod].deviceGrid<float>();
   
    uint32_t leaf_count = voxel_grid->grid_handles[lod].gridMetaData()->nodeCount(0);

    // Upload to GPU
    float* thresholds_gpu;
    checkCuda(cudaMalloc(&thresholds_gpu, cpu_thresholds.size() * sizeof(float)));
    checkCuda(cudaMemcpy(thresholds_gpu, cpu_thresholds.data(), cpu_thresholds.size() * sizeof(float), cudaMemcpyHostToDevice));

    uint32_t* morton_mapping_gpu;
    size_t lod_sparse_voxel_count = voxel_grid->grid_handles[lod].gridMetaData()->activeVoxelCount();

    assert(cpu_thresholds.size() == lod_sparse_voxel_count);
    if (lod == 0) {
        morton_mapping_gpu = morton_sparse_voxel_idx_mapping_gpu;
    } else {
        // Copy temporarily to the gpu
        assert(morton_sparse_voxel_idx_mappings[lod].size() == lod_sparse_voxel_count);
        checkCuda(cudaMalloc(&morton_mapping_gpu,
                             sizeof(uint32_t) * lod_sparse_voxel_count));
        checkCuda(cudaMemcpy(morton_mapping_gpu,
                             morton_sparse_voxel_idx_mappings[lod].data(),
                             sizeof(uint32_t) * lod_sparse_voxel_count,
                             cudaMemcpyHostToDevice));
    }

    float min_threshold = params.threshold_min_value;
    float max_threshold = params.threshold_max_value;

    auto kernel = [grid_d,
                   lod,
                   min_threshold,
                   max_threshold,
                   thresholds_gpu,
                   morton_mapping_gpu,
                   lod_sparse_voxel_count] __device__(const uint64_t n) {
        assert(grid_d->isSequential<0>() == true);
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);  // this only works if grid->isSequential<0>() == true
        const int i = n & 511;

        if (leaf_d->isActive(i)) {
            
            auto coord = coord_to_ivec3(leaf_d->offsetToGlobalCoord(i));
            uint32_t morton_idx = encode_morton3(coord.x, coord.y, coord.z);
            int sparse_idx = -1;
            for (size_t j = 0; j < lod_sparse_voxel_count; j++)
            {
                if(morton_mapping_gpu[j] == morton_idx){
                    sparse_idx = j;
                    break;
                }
            }
            assert(sparse_idx != -1);
            const float new_value = glm::clamp(thresholds_gpu[sparse_idx],min_threshold,max_threshold);
            
            leaf_d->setValueOnly(i, new_value);  // only possible execution divergence
        }
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512 * leaf_count, kernel);

    checkCuda(cudaFree(thresholds_gpu));

    if (lod != 0) {
        checkCuda(cudaFree(morton_mapping_gpu));
    }
}

__global__ void compute_confusion_matrices(uint32_t num_samples, uint32_t num_processed_voxels, uint32_t num_thresholds, float *__restrict__ true_values, float *__restrict__ proba_values, uint32_t* confusion_matrices)
{
    assert(num_thresholds >= 2);

    int threshold_idx = threadIdx.x;
    int i = blockIdx.x;
    
    if (threshold_idx >= num_thresholds || i >= num_samples * num_processed_voxels)
        return;

    ivec2 sample_idx_voxel_offset = coord_1d_to_2d(i, num_samples, num_processed_voxels);
    int voxel_offset = sample_idx_voxel_offset.y;

    assert(i == sample_idx_voxel_offset.x + voxel_offset * num_samples);

    assert(true_values[i] == 1.f || true_values[i] == 0.0f);

    const bool true_val = true_values[i] == 1.0f;
    const float proba_val = proba_values[i];

    const float threshold = threshold_idx / (float)(num_thresholds - 1);
    int cm_idx = coord_2d_to_1d((threshold_idx * 4),voxel_offset, num_thresholds * 4, num_processed_voxels);

    //  confusion_matrix[cm_idx + 0] = TN
    //  confusion_matrix[cm_idx + 1] = FP
    //  confusion_matrix[cm_idx + 2] = FN
    //  confusion_matrix[cm_idx + 3] = TP

    bool predicted_val = proba_val > threshold;

    // For convenience same order as in sklearn python module
    // i.e. [ True Negative, False positive, False Negative, True Positive ]
    if (!true_val) { // Should be classified as non-occluded
        if (!predicted_val)  // TN, it was also classified as non-occluded 
            atomicAdd(&confusion_matrices[cm_idx], 1);
        else  // FP, it was incorrectly classified as occluded when it should be non-occluded
            atomicAdd(&confusion_matrices[cm_idx + 1], 1);
    } else { // Should be classified as occlusion
        if (!predicted_val) // FN, it was incorrectly classified as non-occluded when it should be occluded
            atomicAdd(&confusion_matrices[cm_idx + 2], 1);
        else // TP, it was classified correctly as occluded
            atomicAdd(&confusion_matrices[cm_idx + 3], 1);
    }
}

__global__ void compute_scores(uint32_t num_elements, uint32_t num_thresholds, uint32_t num_processed_voxels, uint32_t* confusion_matrices, float* scores, int metric, float threshold_fscore_beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_elements)
        return;

    ivec2 threshold_idx_voxel_offset = coord_1d_to_2d(i, num_thresholds, num_processed_voxels);
    
    int threshold_idx = threshold_idx_voxel_offset.x;
    int voxel_offset =  threshold_idx_voxel_offset.y;
    assert(voxel_offset < num_processed_voxels);

    int cm_idx = coord_2d_to_1d(threshold_idx * 4, voxel_offset, num_thresholds * 4, num_processed_voxels);

    uint32_t tn = confusion_matrices[cm_idx + 0]; 
    uint32_t fp = confusion_matrices[cm_idx + 1]; 
    uint32_t fn = confusion_matrices[cm_idx + 2]; 
    uint32_t tp = confusion_matrices[cm_idx + 3];

    int score_idx =
        coord_2d_to_1d(threshold_idx, voxel_offset, num_thresholds, num_processed_voxels);

    if(metric == NeuralLodLearning::ThresholdOptimMetrics::Precision)
        scores[score_idx] = precision_score(fp,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::Recall)
        scores[score_idx] = recall_score(fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::FScore)
        scores[score_idx] = fscore(tn,fp,fn,tp,threshold_fscore_beta);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::FScoreMacro)
        scores[score_idx] = fscore_macro(tn,fp,fn,tp,threshold_fscore_beta);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::MCC) 
        scores[score_idx] = mcc(tn,fp,fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::FScoreWeighted){
        int support_cm_idx = coord_2d_to_1d(
            (num_thresholds - 1) * 2, voxel_offset, num_thresholds * 4, num_processed_voxels);
        glm::uvec2 support = glm::uvec2(
            confusion_matrices[support_cm_idx + 3] + confusion_matrices[support_cm_idx + 2],
            confusion_matrices[support_cm_idx + 1] + confusion_matrices[support_cm_idx + 0]);
        scores[score_idx] = fscore_weighted(tn,fp,fn,tp,support,threshold_fscore_beta);
    }else if(metric == NeuralLodLearning::ThresholdOptimMetrics::OptimalFScoreWeighted){
        int support_cm_idx = coord_2d_to_1d(
            (num_thresholds - 1) * 2, voxel_offset, num_thresholds * 4, num_processed_voxels);
        glm::uvec2 support = glm::uvec2(
            confusion_matrices[support_cm_idx + 3] + confusion_matrices[support_cm_idx + 2],
            confusion_matrices[support_cm_idx + 1] + confusion_matrices[support_cm_idx + 0]);
        scores[score_idx] = optimal_fscore_weighted(tn,fp,fn,tp,support,threshold_fscore_beta);
    }else if(metric == NeuralLodLearning::ThresholdOptimMetrics::CohenKappa) 
        scores[score_idx] = cohen_kappa(tn,fp,fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::Accuracy) 
        scores[score_idx] = accuracy(tn,fp,fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::BalancedAccuracy) 
        scores[score_idx] = balanced_accuracy(tn,fp,fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::FowlkesMallows)
        scores[score_idx] = fowlkes_mallows(tn,fp,fn,tp);
    else if(metric == NeuralLodLearning::ThresholdOptimMetrics::YoudenJ){
        scores[score_idx] = youden_j(tn,fp,fn,tp);
        if (scores[score_idx] < -1.f){
            printf("scores[score_idx] : %f\n",scores[score_idx]);
        }
    }
    else
        assert(false);

    return;
}

__global__ void compute_optimal_thresholds(uint32_t num_processed_voxels, uint32_t num_thresholds, uint32_t sparse_voxel_offset, float* scores, float* optimal_thresholds)
{
    int voxel_offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (voxel_offset >= num_processed_voxels)
        return;

    
    float max_score = -10.0f; // All scores in [-1,1]
    int optimal_idx = -1;
    for (size_t threshold_idx = 0; threshold_idx < num_thresholds; threshold_idx++)
    {   
        int score_idx = coord_2d_to_1d(threshold_idx, voxel_offset, num_thresholds, num_processed_voxels);
        const float value = scores[score_idx];
        if(value > max_score){
            max_score = value;
            optimal_idx = threshold_idx;
        }
    }
    assert(optimal_idx != -1);
    optimal_idx = voxel_offset * num_thresholds + optimal_idx;
    
    uint32_t equal_indices = 0;
    while (abs(scores[optimal_idx + equal_indices] - scores[optimal_idx]) < 1e-4 &&
           optimal_idx + equal_indices < (voxel_offset + 1) * num_thresholds) {
        equal_indices++;
    }
    optimal_idx += (equal_indices / 2);

    const float optimal_threshold = (optimal_idx - voxel_offset * num_thresholds) / (float)(num_thresholds - 1);
    assert(optimal_threshold >= 0.f && optimal_threshold <= 1.f);
    optimal_thresholds[sparse_voxel_offset + voxel_offset] = optimal_threshold;    
}

void NeuralLodLearning::optimise_thresholds_for_voxel_grid(ThresholdOptimMetrics metric)
{
    assert(backend->backend->params.ray_query_method == 1);

    println(CLL::WARNING,"Optimising thresholds, this might take some time..");

    int rt_rayquery_index = backend->backend->variant_index("PT");
    cudaStream_t default_stream = 0;

    uint32_t visibility_network_batch_size = params.visibility_nn.batch_size;

    // Todo : remove hardcoded pathfiles
    std::string optimal_thresholds_filename_prefix = "../../neural_lod/results/optimal_thresholds/";

    const uint32_t num_thresholds = 101;
    const uint32_t max_parallel_voxels = params.threshold_max_parralel_voxels;   
    
    size_t min_lod = params.threshold_min_lod;
    size_t max_lod = min(params.threshold_max_lod,NUM_LODS());

    for (size_t lod = min_lod; lod < max_lod; lod++) {
        
        println(CLL::WARNING,"Processing lod : %i",lod);

        size_t lod_sparse_voxel_count = voxel_grid->grid_handles[lod].gridMetaData()->activeVoxelCount();

        uint32_t* morton_mapping_gpu;
        if (lod == 0) {
            morton_mapping_gpu = morton_sparse_voxel_idx_mapping_gpu;
        } else {
            // Copy temporarily to the gpu
            assert(morton_sparse_voxel_idx_mappings[lod].size() == lod_sparse_voxel_count);
            checkCuda(cudaMalloc(&morton_mapping_gpu,
                                sizeof(uint32_t) * lod_sparse_voxel_count));
            checkCuda(cudaMemcpy(morton_mapping_gpu,
                                morton_sparse_voxel_idx_mappings[lod].data(),
                                sizeof(uint32_t) * lod_sparse_voxel_count,
                                cudaMemcpyHostToDevice));
        }

        params.visibility_nn.batch_size = max_parallel_voxels * params.threshold_optim_samples;

        if(params.visibility_nn.batch_size > DEFAULT_RAY_QUERY_BUDGET * 2){
            println(CLL::CRITICAL,
                    "Batch size > %i (DEFAULT_RAY_QUERY_BUDGET * 2), please adapt "
                    "threshold_max_parralel_voxels and threshold_optim_samples adequatly!");
        }
        
        initialize_visibility_nn_data();

        float* optimal_thresholds_managed;
        uint32_t* confusion_matrices;
        float* scores;

        checkCuda(cudaMalloc(&confusion_matrices, max_parallel_voxels * num_thresholds * 4 * sizeof(uint32_t)));
        checkCuda(cudaMalloc(&scores, max_parallel_voxels * num_thresholds * sizeof(float)));
        checkCuda(cudaMallocManaged(&optimal_thresholds_managed, lod_sparse_voxel_count * sizeof(float)));

        // Get N samples per voxel and compute its optimal threshold
        for (int i = 0; i < lod_sparse_voxel_count; i+= max_parallel_voxels) {
            // 4 elements per matrix
            checkCuda(cudaMemset(confusion_matrices,0, max_parallel_voxels * num_thresholds * 4 * sizeof(uint32_t)));
            checkCuda(cudaMemset(scores, 0, max_parallel_voxels * num_thresholds * sizeof(float)));

            uint32_t num_processed_voxels = max_parallel_voxels;
            // last iteration possibly less voxels processed
            if(i + max_parallel_voxels > lod_sparse_voxel_count){
                num_processed_voxels = -(i - lod_sparse_voxel_count);
                assert(num_processed_voxels < max_parallel_voxels);
            }

            assert(max_parallel_voxels * params.threshold_optim_samples == params.visibility_nn.batch_size);
            // Compute reference values at random coordinates
            tcnn::linear_kernel(neural_lod_learning::fill_ray_queries_visibility,
                                0,
                                default_stream,
                                params.visibility_nn.batch_size,
                                num_processed_voxels, // the number of voxels we process simultaneously
                                morton_mapping_gpu,
                                params.threshold_optim_samples, // the number of samples per voxel
                                ivec3(i),
                                backend->ray_queries,
                                visibility_nn->inference_input_batch.data(),
                                nullptr,
                                nullptr,
                                params.visibility_nn.voxel_extent_dilation_outward,
                                params.visibility_nn.voxel_extent_dilation_inward,
                                params.visibility_nn.voxel_extent_min_dilation,
                                params.visibility_nn.voxel_bound_bias,
                                false,
                                false,
                                0.0,
                                0.0,
                                lod);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            backend->backend->render_ray_queries(
                params.visibility_nn.batch_size, backend->backend->params, rt_rayquery_index);
            checkCuda(cudaDeviceSynchronize());

            tcnn::linear_kernel(neural_lod_learning::record_ray_query_results_visibility_optim,
                                0,
                                default_stream,
                                params.visibility_nn.batch_size,
                                backend->ray_results,
                                visibility_nn->training_target_batch.data());
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            // For those sampled inputs, do an inference now
            visibility_nn->network->inference(default_stream,
                                              visibility_nn->inference_input_batch,
                                              visibility_nn->prediction);
            // Ensure we have exactly 1 row
            assert(visibility_nn->prediction.rows() == 1);
            
            assert(num_thresholds == 101);
            dim3 dimBlock(num_thresholds); // each thread in a block computes `processed_voxels` different threshold
            dim3 dimGrid(params.threshold_optim_samples * num_processed_voxels);

            // // Compute the confusion matrix for num_thresholds elements
            compute_confusion_matrices<<<dimGrid, dimBlock>>>(
                params.threshold_optim_samples,
                num_processed_voxels,
                num_thresholds,
                visibility_nn->training_target_batch.data(),
                visibility_nn->prediction.data(),
                confusion_matrices);
            // checkCuda(cudaGetLastError());
            // checkCuda(cudaDeviceSynchronize());
                
            tcnn::linear_kernel(compute_scores,
                                0,
                                default_stream,
                                num_thresholds * num_processed_voxels,
                                num_thresholds,
                                num_processed_voxels,
                                confusion_matrices,
                                scores,
                                (int)metric,
                                params.threshold_fscore_beta);
            // checkCuda(cudaGetLastError());
            // checkCuda(cudaDeviceSynchronize());

            tcnn::linear_kernel(compute_optimal_thresholds,
                                0,
                                default_stream,
                                num_processed_voxels,
                                num_thresholds,
                                i,
                                scores,
                                optimal_thresholds_managed);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());        
        }

        // Free the confusion matrix when it is no longer needed
        checkCuda(cudaFree(confusion_matrices));
        checkCuda(cudaFree(scores));

        // Finally write the optimised thresholds to each voxel
        {
            auto grid_d = voxel_grid->grid_handles[lod].deviceGrid<float>();

            uint32_t leaf_count = voxel_grid->grid_handles[lod].gridMetaData()->nodeCount(0);

            float min_threshold = params.threshold_min_value;
            float max_threshold = params.threshold_max_value;

            auto kernel = [grid_d,
                           lod,
                           min_threshold,
                           max_threshold,
                           optimal_thresholds_managed,
                           morton_mapping_gpu,
                           lod_sparse_voxel_count] __device__(const uint64_t n) {
                assert(grid_d->isSequential<0>() == true);
                auto *leaf_d = grid_d->tree().getFirstNode<0>() +
                               (n >> 9);  // this only works if grid->isSequential<0>() == true
                const int i = n & 511;

                if (leaf_d->isActive(i)) {
                    auto coord = coord_to_ivec3(leaf_d->offsetToGlobalCoord(i));
                    uint32_t morton_idx = encode_morton3(coord.x, coord.y, coord.z);
                    int sparse_idx = -1;
                    for (size_t j = 0; j < lod_sparse_voxel_count; j++) {
                        if (morton_mapping_gpu[j] == morton_idx) {
                            sparse_idx = j;
                            break;
                        }
                    }
                    assert(sparse_idx != -1);
                    const float new_value = glm::clamp(optimal_thresholds_managed[sparse_idx], min_threshold, max_threshold);

                    leaf_d->setValueOnly(i, new_value);  // only possible execution divergence
                }
            };

            thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
            thrust::for_each(iter, iter + 512 * leaf_count, kernel);
        }

        checkCuda(cudaDeviceSynchronize());
        // Use the managed memory as a vector to write to disk
        std::vector<float> data(optimal_thresholds_managed, optimal_thresholds_managed + lod_sparse_voxel_count);

        // Write optimal thresholds to disk
        auto filename = weights_filename_prefix != "" ? weights_filename_prefix : scene_name;
        auto optimal_threshold_value_file = optimal_thresholds_filename_prefix + filename + "_lod_" + std::to_string(lod) + "_optimal_thresholds.bin";
        write_vector_to_file(data,optimal_threshold_value_file);

        if (lod != 0) {
            checkCuda(cudaFree(morton_mapping_gpu));
        }
        checkCuda(cudaFree(optimal_thresholds_managed));
    }
    

    // Set back batch size
    params.visibility_nn.batch_size = visibility_network_batch_size;
    
    initialize_visibility_nn_data();

    println(CLL::WARNING,"Done!");
}

void NeuralLodLearning::throughput_network_one_step_optim()
{
    cudaStream_t tiny_nn_stream = backend->concurrent_streams[0];
    cudaStream_t default_stream = 0;

    int rt_rayquery_index = backend->backend->variant_index("PT");
    // one step optimisation here
    if (throughput_nn->training_step < params.throughput_nn.n_training_steps) {
        bool print_loss = throughput_nn->training_step % 100 == 0;
        recompute_inference = params.compute_inference_during_training &&
                              throughput_nn->training_step % 100 == 0;

        {
            // Compute reference values at random coordinates
            tcnn::linear_kernel(
                neural_lod_learning::fill_ray_queries_throughput,
                0,
                default_stream,
                params.throughput_nn.batch_size,
                sparse_voxel_count,
                morton_sparse_voxel_idx_mapping_gpu,
                throughput_nn->training_step,
                params.learn_single_voxel ? params.selected_voxel_idx : glm::ivec3(-1),
                backend->ray_queries,
                throughput_nn->training_input_batch.data(),
                inferences_sample_count_grid,
                inference_total_sample_count,
                params.visibility_nn.voxel_extent_dilation_outward,
                params.visibility_nn.voxel_extent_dilation_inward,
                params.visibility_nn.voxel_extent_min_dilation,
                params.visibility_nn.voxel_bound_bias,
                params.max_throughput_depth,
                throughput_nn->training_step,
                params.throughput_nn.pdf_strength,
                params.throughput_nn.pdf_shift);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            backend->backend->render_ray_queries(
                params.throughput_nn.batch_size, backend->backend->params, rt_rayquery_index);
            checkCuda(cudaDeviceSynchronize());

            int *invalid_samples;
            checkCuda(cudaMallocManaged(&invalid_samples, sizeof(int)));
            *invalid_samples = 0;

            tcnn::linear_kernel(neural_lod_learning::record_ray_query_results_throughput_optim,
                                0,
                                default_stream,
                                params.throughput_nn.batch_size,
                                backend->ray_results,
                                throughput_nn->training_target_batch.data(),
                                throughput_nn->training_input_batch.data(),
                                invalid_samples,
                                params.throughput_nn.log_learning);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            uint32_t max_valid_batch_samples =
                tcnn::next_multiple(params.throughput_nn.batch_size - *invalid_samples,
                                    tcnn::batch_size_granularity) -
                tcnn::batch_size_granularity;

            reinitialize_object(
                throughput_nn->valid_training_input_batch,
                tcnn::GPUMatrix<float>(throughput_n_input_dims, max_valid_batch_samples));
            reinitialize_object(
                throughput_nn->valid_training_target_batch,
                tcnn::GPUMatrix<float>(throughput_n_output_dims, max_valid_batch_samples));

            int *valid_samples;
            checkCuda(cudaMallocManaged(&valid_samples, sizeof(int)));
            *valid_samples = 0;

            tcnn::linear_kernel(neural_lod_learning::copy_valid_results_throughput_optim,
                                0,
                                default_stream,
                                params.throughput_nn.batch_size,
                                valid_samples,
                                throughput_nn->training_target_batch.data(),
                                throughput_nn->training_input_batch.data(),
                                throughput_nn->valid_training_target_batch.data(),
                                throughput_nn->valid_training_input_batch.data(),
                                max_valid_batch_samples);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            assert(*valid_samples == max_valid_batch_samples && *valid_samples != 0);

            checkCuda(cudaFree(invalid_samples));
            checkCuda(cudaFree(valid_samples));
        }

        // Training step
        {
            auto ctx = throughput_nn->trainer->training_step(
                tiny_nn_stream,
                throughput_nn->valid_training_input_batch,
                throughput_nn->valid_training_target_batch);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            throughput_nn->tmp_loss += throughput_nn->trainer->loss(tiny_nn_stream, *ctx);
            ++throughput_nn->tmp_loss_counter;
        }

        // Debug outputs
        {
            if (print_loss) {
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                std::cout << "Throughput Step #" << throughput_nn->training_step << ": "
                          << "loss="
                          << throughput_nn->tmp_loss / (float)throughput_nn->tmp_loss_counter
                          << " time="
                          << float(std::chrono::duration_cast<std::chrono::microseconds>(
                                       end - throughput_nn->begin_time)
                                       .count()) *
                                 1e-6
                          << "[s]" << std::endl;

                throughput_nn->tmp_loss = 0;
                throughput_nn->tmp_loss_counter = 0;
            }

            // Don't count visualizing as part of timing
            if (print_loss) {
                throughput_nn->begin_time = std::chrono::steady_clock::now();
            }
        }

        ++throughput_nn->training_step;
    }
}

void NeuralLodLearning::visibility_network_one_step_optim()
{
    cudaStream_t tiny_nn_stream = backend->concurrent_streams[0];
    cudaStream_t default_stream = 0;

    int rt_rayquery_index = backend->backend->variant_index("PT");
    // one step optimisation here
    if (visibility_nn->training_step < params.visibility_nn.n_training_steps) {
        bool print_loss = visibility_nn->training_step % 100 == 0;
        recompute_inference = params.compute_inference_during_training &&
                                visibility_nn->training_step % 100 == 0;
 
        {
            // Compute reference values at random coordinates
            tcnn::linear_kernel(
                neural_lod_learning::fill_ray_queries_visibility,
                0,
                default_stream,
                params.visibility_nn.batch_size,
                sparse_voxel_count,
                morton_sparse_voxel_idx_mapping_gpu,
                visibility_nn->training_step,
                params.learn_single_voxel ? params.selected_voxel_idx : glm::ivec3(-1),
                backend->ray_queries,
                visibility_nn->training_input_batch.data(),
                inferences_sample_count_grid,
                inference_total_sample_count,
                params.visibility_nn.voxel_extent_dilation_outward,
                params.visibility_nn.voxel_extent_dilation_inward,
                params.visibility_nn.voxel_extent_min_dilation,
                params.visibility_nn.voxel_bound_bias,
                params.visibility_nn.increase_boundary_density,
                false,
                params.visibility_nn.pdf_strength,
                params.visibility_nn.pdf_shift,
                -1);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            backend->backend->render_ray_queries(params.visibility_nn.batch_size,
                                                    backend->backend->params,
                                                    rt_rayquery_index);
            checkCuda(cudaDeviceSynchronize());

            tcnn::linear_kernel(
                neural_lod_learning::record_ray_query_results_visibility_optim,
                0,
                default_stream,
                params.visibility_nn.batch_size,
                backend->ray_results,
                visibility_nn->training_target_batch.data());
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }

        // Training step
        {
            auto ctx = visibility_nn->trainer->training_step(
                tiny_nn_stream,
                visibility_nn->training_input_batch,
                visibility_nn->training_target_batch);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            visibility_nn->tmp_loss +=
                visibility_nn->trainer->loss(tiny_nn_stream, *ctx);
            ++visibility_nn->tmp_loss_counter;
        }

        // Debug outputs
        {
            if (print_loss) {
                std::chrono::steady_clock::time_point end =
                    std::chrono::steady_clock::now();
                std::cout
                    << "Visibility Step #" << visibility_nn->training_step << ": "
                    << "loss="
                    << visibility_nn->tmp_loss / (float)visibility_nn->tmp_loss_counter
                    << " time="
                    << float(std::chrono::duration_cast<std::chrono::microseconds>(
                                    end - visibility_nn->begin_time)
                                    .count()) *
                            1e-6
                    << "[s]" << std::endl;

                visibility_nn->tmp_loss = 0;
                visibility_nn->tmp_loss_counter = 0;
            }

            // Don't count visualizing as part of timing
            if (print_loss) {
                visibility_nn->begin_time = std::chrono::steady_clock::now();
            }
        }

        ++visibility_nn->training_step;
    }
}

void NeuralLodLearning::render()
{
    if (!scene_loaded || !throughput_nn || !visibility_nn)
        return;

    checkCuda(cudaMemcpyToSymbol(framebuffer, &backend->pixel_buffer(), sizeof(framebuffer)));
    checkCuda(cudaMemcpyToSymbol(accum_buffer, &backend->accum_buffer(), sizeof(accum_buffer)));
    checkCuda(cudaMemcpyToSymbol(debug_buffer, &backend->debug_buffer, sizeof(debug_buffer)));
    checkCuda(cudaMemcpyToSymbol(fb_width, &backend->screen_width, sizeof(fb_width)));
    checkCuda(cudaMemcpyToSymbol(fb_height, &backend->screen_height, sizeof(fb_height)));

    checkCuda(cudaDeviceSynchronize());

    // Add concurent streams?
    cudaStream_t tiny_nn_stream = backend->concurrent_streams[0];
    cudaStream_t default_stream = 0;
    auto camera = backend->backend->camera;

    int rt_rayquery_index = backend->backend->variant_index("PT");

    // Make sure we are in a consistent mode for the ray queries
    int ray_query_method = backend->backend->params.ray_query_method;
    if ( params.optimisation_type == OptimisationType::ThroughputOptim && ray_query_method != 0) {
        // Restart in correct mode
        backend->backend->params.ray_query_method = 0;
        return;
    }
    if ((params.optimisation_type == OptimisationType::VisibilityOptim ||
         params.optimisation_type == OptimisationType::ThresholdsOptim) &&
        ray_query_method != 1) {
        // Restart in correct mode
        backend->backend->params.ray_query_method = 1;
        return;
    }

    /**** Optimization *****/

    if (lod_grids_generated && !params.compute_optimisation_ref && params.run_optimisation){
        if(params.optimisation_type == OptimisationType::ThroughputOptim) {
            throughput_network_one_step_optim();
        }else if (params.optimisation_type == OptimisationType::VisibilityOptim){
            visibility_network_one_step_optim();
        }
        else if (params.optimisation_type == OptimisationType::ThresholdsOptim){
            optimise_thresholds_for_voxel_grid(params.threshold_optim_metric);
            params.run_optimisation = false;
        }
    }

    /***** Both Visualization and Optimisation Inference *****/
    if (recompute_inference) {
        if (params.optimisation_type == OptimisationType::ThroughputOptim) {
            initialize_throughput_nn_data();
            throughput_nn->network->inference(tiny_nn_stream,
                                                throughput_nn->inference_input_batch,
                                                throughput_nn->prediction);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            assert(throughput_nn->prediction.rows() == throughput_n_output_dims);
            recompute_inference = false;
        }
    }

    /***** Visualization  *****/

    if (params.debug_view == DebugView::VoxelGridView) {
        uint32_t frame_buffer_elements = backend->screen_width * backend->screen_height;

        // Voxelisation specific visualisation
        if (voxel_grid->grid_handles[current_lod].grid<float>() != nullptr) {
            tcnn::linear_kernel(raymarch_grid,
                                0,
                                default_stream,
                                frame_buffer_elements,
                                camera.pos,
                                camera.dir,
                                camera.up,
                                float(backend->screen_width) / float(backend->screen_height),
                                voxel_grid->grid_handles[current_lod].deviceGrid<float>(),
                                params.debug_grid_data);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }

    } else if (params.debug_view == DebugView::ThroughputView) {
        /***** Reference Computation for visualisation *****/

        if(params.compute_optimisation_ref){
            // Compute reference values at random coordinates
            tcnn::linear_kernel(
                neural_lod_learning::fill_ray_queries_throughput,
                0,
                default_stream,
                params.throughput_nn.batch_size,
                sparse_voxel_count,
                morton_sparse_voxel_idx_mapping_gpu,
                batch_index,
                params.selected_voxel_idx,  // Always the selected index for visualisation
                backend->ray_queries,
                nullptr,
                nullptr,
                nullptr,
                params.visibility_nn.voxel_extent_dilation_outward,
                params.visibility_nn.voxel_extent_dilation_inward,
                params.visibility_nn.voxel_extent_min_dilation,
                params.visibility_nn.voxel_bound_bias,
                params.max_throughput_depth,
                -1,
                params.throughput_nn.pdf_strength,
                params.throughput_nn.pdf_shift);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            backend->backend->render_ray_queries(params.throughput_nn.batch_size,
                                                    backend->backend->params,
                                                    rt_rayquery_index);

            tcnn::linear_kernel(neural_lod_learning::record_reference_throughput_estimates,
                                0,
                                default_stream,
                                params.throughput_nn.batch_size,
                                backend->ray_queries,
                                backend->ray_results,
                                throughput_nn->ref_output_accumulated.data(),
                                params.throughput_nn.inference_2d_sampling_res,
                                params.throughput_nn.log_learning);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }

        if(lod_grids_generated){
             // Optimisation specific visualisation

            uint32_t frame_buffer_elements = backend->screen_width * backend->screen_height;

            float *data = params.show_optimisation_ref
                              ? throughput_nn->ref_output_accumulated.data()
                              : throughput_nn->prediction.data();

            tcnn::linear_kernel(fill_debug_framebuffer_with_throughput_data,
                                0,
                                default_stream,
                                frame_buffer_elements,
                                data,
                                params.throughput_nn.inference_2d_sampling_res,
                                true,
                                params.show_optimisation_ref,
                                params.throughput_nn.log_learning);

            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }
        ++batch_index;
    }
}

NEURAL_LOD_LEARNING_NAMESPACE_END