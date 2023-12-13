#include <tiny-cuda-nn/common.h>

#include "neural_lod_learning.h"
#include "neural_lod_learning_common.cuh"
#include "neural_lod_learning_grid.cuh"

#include "../../imstate.h"
#include "libapp/imutils.h"

#include "../util/interactive_camera.h"
#include "vulkan/render_vulkan.h"

NEURAL_LOD_LEARNING_NAMESPACE_BEGIN

bool encoding_ui(std::string encoding_label,
                 std::string network_label,
                 optimconfig::EncodingType &encoding_type,
                 optimconfig::GridOptions &grid_options,
                 optimconfig::ShOptions *sh_options = nullptr)
{
    bool model_changed = false;

    int last_active = valid_combo_index(encoding_type, encoding_types);
    if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                          (encoding_label + " Encoding##" + network_label).c_str(),
                          encoding_types,
                          encoding_types[last_active])) {
        for (int i = 0; i < ARRAYSIZE(encoding_types); ++i) {
            if (IMGUI_STATE(ImGui::Selectable, encoding_types[i], i == last_active)) {
                encoding_type = (EncodingType)i;

                model_changed = true;
            }
        }

        IMGUI_STATE_END(ImGui::EndCombo, encoding_types);
    }

    if (encoding_type == EncodingType::HashGrid || encoding_type == EncodingType::DenseGrid ||
        encoding_type == EncodingType::TiledGrid) {
        if (IMGUI_STATE_BEGIN_HEADER(
                ImGui::CollapsingHeader,
                (encoding_label + " Grid Options##" + network_label).c_str(),
                &grid_options,
                ImGuiTreeNodeFlags_DefaultOpen)) {
            model_changed |=
                IMGUI_STATE(ImGui::InputInt,
                            ("n_levels##" + encoding_label + network_label).c_str(),
                            &grid_options.n_levels);
            model_changed |= IMGUI_STATE(
                ImGui::InputInt,
                ("n_features_per_level##" + encoding_label + network_label).c_str(),
                &grid_options.n_features_per_level);
            model_changed |=
                IMGUI_STATE(ImGui::InputInt,
                            ("log2_hashmap_size##" + encoding_label + network_label).c_str(),
                            &grid_options.log2_hashmap_size);
            model_changed |=
                IMGUI_STATE(ImGui::InputFloat,
                            ("per_level_scale##" + encoding_label + network_label).c_str(),
                            &grid_options.per_level_scale);
            model_changed |=
                IMGUI_STATE(ImGui::InputInt,
                            ("base_resolution##" + encoding_label + network_label).c_str(),
                            &grid_options.base_resolution);

            model_changed |= IMGUI_STATE(
                ImGui::Checkbox,
                ("stochastic_interpolation##" + encoding_label + network_label).c_str(),
                &grid_options.stochastic_interpolation);

            last_active = valid_combo_index(grid_options.interpolation, interpolation_types);
            if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                  ("Interpolation##" + encoding_label + network_label).c_str(),
                                  interpolation_types,
                                  interpolation_types[last_active])) {
                for (int i = 0; i < ARRAYSIZE(interpolation_types); ++i) {
                    if (IMGUI_STATE(
                            ImGui::Selectable, interpolation_types[i], i == last_active)) {
                        grid_options.interpolation = (InterpolationType)i;
                        model_changed = true;
                    }
                }
                IMGUI_STATE_END(ImGui::EndCombo, interpolation_types);
            }
        }
    } else if (sh_options && encoding_type == EncodingType::SphericalHarmonics) {
        model_changed |=
            IMGUI_STATE(ImGui::InputInt,
                        ("Max SH degree##" + encoding_label + network_label).c_str(),
                        &sh_options->max_degree);
    }

    return model_changed;
}

bool NeuralLodLearning::ui_and_state(bool &renderer_changed)
{
    auto debug_texture_desc_set = ((RenderVulkan *)backend->backend)->debug_texture_desc_set;

    int last_active = current_lod;
    if (IMGUI_STATE_BEGIN(
            ImGui::BeginCombo, "current lod", lod_levels, lod_levels[last_active])) {
        for (uint32_t i = 0; i < NUM_LODS(); ++i)
            if (IMGUI_STATE(ImGui::Selectable, lod_levels[i], i == last_active)) {
                set_lod(i);
            }
        IMGUI_STATE_END(ImGui::EndCombo, lod_levels);
    }
    IMGUI_VOLATILE(
        ImGui::Text("Resolution of LoD : %i^3", VOXEL_GRID_MAX_RES() >> current_lod));
    IMGUI_SPACE_SEPARATOR();

    if (IMGUI_STATE_BEGIN_ALWAYS(
            ImGui::BeginTabBar, "Neural Learning Options", debug_texture_desc_set)) {
        if (IMGUI_STATE_BEGIN(ImGui::BeginTabItem, "Debug View", debug_texture_desc_set)) {
            int last_active = valid_combo_index(params.debug_view, debug_views);
            if (IMGUI_STATE_BEGIN(
                    ImGui::BeginCombo, "Debug view", debug_views, debug_views[last_active])) {
                for (int i = 0; i < ARRAYSIZE(debug_views); ++i) {
                    if (IMGUI_STATE(ImGui::Selectable, debug_views[i], i == last_active)) {
                        params.debug_view = (DebugView)i;
                        needs_rerender = true;
                        if (params.debug_view == DebugView::ThroughputView) {
                            params.optimisation_type = OptimisationType::ThroughputOptim;
                        }
                        recompute_inference = true;
                    }
                }
                IMGUI_STATE_END(ImGui::EndCombo, debug_views);
            }

            IMGUI_SPACE_SEPARATOR();

            if (params.debug_view == DebugView::ThroughputView) {
                params.square_debug_view = true;

                if (IMGUI_STATE_ACTION(ImGui::Button, "Restart Computing Optimisation ref")) {
                    params.compute_optimisation_ref = true;
                    recompute_inference = true;
                }

                if (params.compute_optimisation_ref &&
                    IMGUI_STATE_ACTION(ImGui::Button, "Stop Computing Optimisation ref")) {
                    params.compute_optimisation_ref = false;
                } else if (!params.compute_optimisation_ref &&
                           IMGUI_STATE_ACTION(ImGui::Button,
                                              "Continue Computing Optimisation ref")) {
                    params.compute_optimisation_ref = true;
                }

                IMGUI_STATE(
                    ImGui::Checkbox, "show optimisation ref", &params.show_optimisation_ref);

                if (IMGUI_STATE3(ImGui::SliderInt,
                                 "selected voxel index (xyz)",
                                 &params.selected_voxel_idx.x,
                                 0,
                                 (VOXEL_GRID_MAX_RES() >> current_lod) - 1)) {
                    recompute_inference = true;
                };

                if (IMGUI_STATE_ACTION(ImGui::Button, "next sparse voxel")) {
                    uint32_t encoded_morton = encode_morton3(params.selected_voxel_idx.x,
                                                             params.selected_voxel_idx.y,
                                                             params.selected_voxel_idx.z);
                    auto next =
                        std::upper_bound(morton_sparse_voxel_idx_mappings[current_lod].begin(),
                                         morton_sparse_voxel_idx_mappings[current_lod].end(),
                                         encoded_morton);
                    params.selected_voxel_idx =
                        next != morton_sparse_voxel_idx_mappings[current_lod].end()
                            ? decode_morton3(*next)
                            : decode_morton3(morton_sparse_voxel_idx_mappings[current_lod][0]);
                    recompute_inference = true;
                }

            } else if (params.debug_view == DebugView::VoxelGridView) {
                IMGUI_STATE1(ImGui::Checkbox, "Show grid data", &params.debug_grid_data);

                params.square_debug_view = false;
            } else {
                params.square_debug_view = false;
            }

            IMGUI_SPACE_SEPARATOR();

            if (ImState::InDefaultMode()) {
                ImVec2 avail_size = ImGui::GetContentRegionAvail();
                ImVec2 im_size;
                if (params.square_debug_view) {
                    float min_side = std::min(avail_size.x, avail_size.y);
                    im_size = ImVec2(min_side, min_side);
                } else {
                    float scaling = std::min(avail_size.x / backend->screen_width,
                                             avail_size.y / backend->screen_height);
                    im_size = ImVec2(backend->screen_width * scaling,
                                     backend->screen_height * scaling);
                }
                ImGui::Image((ImTextureID)debug_texture_desc_set,
                             im_size,
                             ImVec2(0, 0),
                             ImVec2(1, 1),
                             ImVec4(1, 1, 1, 1),
                             ImVec4(1, 1, 1, 1));
            };

            IMGUI_STATE_END(ImGui::EndTabItem, debug_texture_desc_set);
        }

        if (IMGUI_STATE_BEGIN(ImGui::BeginTabItem, "Optim Data", &weights_directory)) {
            IMGUI_STATE_(
                ImGui::InputText, "Weights directory", weights_directory, MAX_FILENAME_SIZE);
            IMGUI_STATE_(ImGui::InputText,
                         "Weights filename prefix",
                         weights_filename_prefix,
                         MAX_FILENAME_SIZE);
            // reinterpret_cast necessary because the class is incomplete at compilation (and
            // so we avoid moving the entire thing to a .cc file instead)
            auto throughput_network =
                reinterpret_cast<neural_lod_learning::NeuralLodLearning::NeuralNet *>(
                    throughput_nn.get());
            auto visibility_network =
                reinterpret_cast<neural_lod_learning::NeuralLodLearning::NeuralNet *>(
                    visibility_nn.get());
            auto complete_prefix = get_complete_weights_prefix_path();

            if (IMGUI_STATE_ACTION(ImGui::Button, "Load throughput weights from disk")) {
                load_weights_from_disk(complete_prefix + "_throughput_weights.json",
                                       throughput_network);
            }
            if (IMGUI_STATE_ACTION(ImGui::Button, "Load visibility weights from disk")) {
                load_weights_from_disk(complete_prefix + "_visibility_weights.json",
                                       visibility_network);
            }

            IMGUI_SPACE_SEPARATOR();

            if (IMGUI_STATE_ACTION(ImGui::Button, "Write throughput weights to disk")) {
                write_weights_to_disk(complete_prefix + "_throughput_weights.json",
                                      throughput_network);
            }
            if (IMGUI_STATE_ACTION(ImGui::Button, "Write visibility weights to disk")) {
                write_weights_to_disk(complete_prefix + "_visibility_weights.json",
                                      visibility_network);
            }

            IMGUI_SPACE_SEPARATOR();

            if (IMGUI_STATE_ACTION(ImGui::Button, "Write throughput config to disk")) {
                write_config_to_disk(complete_prefix + "_throughput_config.json",
                                     throughput_network);
            }
            if (IMGUI_STATE_ACTION(ImGui::Button, "Write visibility config to disk")) {
                write_config_to_disk(complete_prefix + "_visibility_config.json",
                                     throughput_network);
            }

            IMGUI_STATE_END(ImGui::EndTabItem, &weights_directory);
        }
        if (IMGUI_STATE_BEGIN(
                ImGui::BeginTabItem, "Threshold Optim", &params.threshold_optim_samples)) {
            IMGUI_STATE(ImGui::InputInt,
                        "threshold optimisation samples",
                        (int *)&params.threshold_optim_samples);

            IMGUI_STATE(ImGui::InputInt,
                        "threshold optim parralel voxels",
                        (int *)&params.threshold_max_parralel_voxels);

            IMGUI_STATE(ImGui::SliderInt,
                        "min lod processed",
                        (int *)&params.threshold_min_lod,
                        0,
                        params.threshold_max_lod);

            IMGUI_STATE(ImGui::SliderInt,
                        "max lod processed",
                        (int *)&params.threshold_max_lod,
                        params.threshold_min_lod,
                        NUM_LODS());

            if (IMGUI_STATE_ACTION(ImGui::Button, "Optimise Thresholds")) {
                params.run_optimisation = true;
                params.optimisation_type = OptimisationType::ThresholdsOptim;
                params.compute_optimisation_ref = false;
            }

            int last_active =
                valid_combo_index(params.threshold_optim_metric, threshold_optim_metrics);
            if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                  "Threshold Optim Metric",
                                  threshold_optim_metrics,
                                  threshold_optim_metrics[last_active])) {
                for (int i = 0; i < ARRAYSIZE(threshold_optim_metrics); ++i) {
                    if (IMGUI_STATE(
                            ImGui::Selectable, threshold_optim_metrics[i], i == last_active)) {
                        params.threshold_optim_metric = (ThresholdOptimMetrics)i;
                    }
                }
                IMGUI_STATE_END(ImGui::EndCombo, threshold_optim_metrics);
            }

            if (params.threshold_optim_metric == ThresholdOptimMetrics::FScore ||
                params.threshold_optim_metric == ThresholdOptimMetrics::FScoreMacro ||
                params.threshold_optim_metric == ThresholdOptimMetrics::FScoreWeighted ||
                params.threshold_optim_metric ==
                    ThresholdOptimMetrics::OptimalFScoreWeighted) {
                IMGUI_STATE(ImGui::SliderFloat,
                            "F-Score beta value",
                            &params.threshold_fscore_beta,
                            0.0001f,
                            100.f);
            }

            IMGUI_STATE(ImGui::InputFloat, "min threshold", &params.threshold_min_value);
            IMGUI_STATE(ImGui::InputFloat, "max threshold", &params.threshold_max_value);

            IMGUI_SPACE_SEPARATOR();

            if (IMGUI_STATE_ACTION(ImGui::Button, "Update grid with fixed thresholds")) {
                update_grid_with_fixed_thresholds();
            }

            IMGUI_STATE(ImGui::SliderFloat,
                        "threshold fixed value",
                        &params.threshold_fixed_value,
                        0.f,
                        1.f);

            IMGUI_STATE_END(ImGui::EndTabItem, &weights_directory);
        }
        if (IMGUI_STATE_BEGIN(
                ImGui::BeginTabItem, "Throughput Optim", &params.throughput_nn)) {
            if (lod_grids_generated) {
                if (params.optimisation_type == OptimisationType::ThroughputOptim &&
                    params.run_optimisation) {
                    auto str = string_format(
                        "Optimising throughput for max resolution %u and %u LoD levels "
                        "(lowest res is %u)",
                        VOXEL_GRID_MAX_RES(),
                        NUM_LODS(),
                        VOXEL_GRID_MIN_RES());
                    IMGUI_VOLATILE(ImGui::Text("%s", str.c_str()));
                }

                IMGUI_STATE1(ImGui::Checkbox,
                            "Compute Inference during training",
                            &params.compute_inference_during_training);

                IMGUI_STATE(ImGui::SliderInt,
                            "throughput batch size",
                            (int *)&params.throughput_nn.batch_size,
                            tcnn::batch_size_granularity,
                            DEFAULT_RAY_QUERY_BUDGET,
                            "%d",
                            ImGuiSliderFlags_Logarithmic);

                IMGUI_SPACE_SEPARATOR();

                IMGUI_STATE(ImGui::SliderFloat,
                            "pdf strength",
                            &params.throughput_nn.pdf_strength,
                            0.f,
                            1000.f);

                IMGUI_STATE(ImGui::SliderFloat,
                            "pdf shift",
                            &params.throughput_nn.pdf_shift,
                            -0.5f,
                            0.5f);

                if (IMGUI_STATE1(ImGui::Checkbox,
                                "learn log space throughput",
                                &params.throughput_nn.log_learning)) {
                    throughput_model_changed = true;
                }

                IMGUI_SPACE_SEPARATOR();

                IMGUI_STATE(ImGui::InputInt,
                            "throughput max training steps",
                            (int *)&params.throughput_nn.n_training_steps);

                if (IMGUI_STATE_ACTION(ImGui::Button, "Restart Throughput Optimization")) {
                    params.run_optimisation = true;
                    params.optimisation_type = OptimisationType::ThroughputOptim;
                    throughput_model_changed = true;
                    params.compute_optimisation_ref = false;
                }

                if (params.optimisation_type == OptimisationType::ThroughputOptim &&
                    params.run_optimisation &&
                    IMGUI_STATE_ACTION(ImGui::Button, "Pause Throughput Optimization")) {
                    params.run_optimisation = false;
                } else if (params.optimisation_type == OptimisationType::ThroughputOptim &&
                           !params.run_optimisation &&
                           IMGUI_STATE_ACTION(ImGui::Button,
                                              "Continue Throughput Optimization")) {
                    params.run_optimisation = true;
                    params.compute_optimisation_ref = false;
                }

                throughput_model_changed |= IMGUI_STATE(ImGui::SliderInt,
                                                        "max throughput depth",
                                                        &params.max_throughput_depth,
                                                        2,
                                                        1000);

                IMGUI_STATE1(ImGui::Checkbox, "learn single voxel", &params.learn_single_voxel);
            }else{
                // Hacky : The else is necessary to properly load the log learning variable from the config file
                if (IMGUI_STATE1(ImGui::Checkbox,
                                "learn log space throughput",
                                &params.throughput_nn.log_learning)) {
                    throughput_model_changed = true;
                }
            }

            if (IMGUI_STATE_BEGIN_HEADER(ImGui::CollapsingHeader,
                                         "Throughput Neural Net Options",
                                         &params.throughput_nn)) {
                throughput_model_changed |=
                    IMGUI_STATE(ImGui::InputInt,
                                "decay start",
                                &params.throughput_nn.optimizer_options.decay_start,
                                0,
                                20000);

                throughput_model_changed |=
                    IMGUI_STATE(ImGui::InputInt,
                                "decay interval",
                                &params.throughput_nn.optimizer_options.decay_interval,
                                1,
                                10000);

                throughput_model_changed |=
                    IMGUI_STATE(ImGui::InputFloat,
                                "decay base",
                                &params.throughput_nn.optimizer_options.decay_base,
                                0.0f,
                                1.0f);

                int last_active =
                    valid_combo_index(params.throughput_nn.optimizer, optimizer_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Optimizer",
                                      optimizer_types,
                                      optimizer_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(optimizer_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, optimizer_types[i], i == last_active)) {
                            params.throughput_nn.optimizer = (OptimizerType)i;
                            throughput_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, optimizer_types);
                }
                if (IMGUI_STATE_BEGIN_HEADER(ImGui::CollapsingHeader,
                                             "Optimizer Options",
                                             &params.throughput_nn.optimizer_options)) {
                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "learning_rate",
                                    &params.throughput_nn.optimizer_options.learning_rate,
                                    0.0f,
                                    1.0f);

                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "beta1",
                                    &params.throughput_nn.optimizer_options.beta1,
                                    0.0f,
                                    1.0f);

                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "beta2",
                                    &params.throughput_nn.optimizer_options.beta2,
                                    0.0f,
                                    1.0f);

                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "epsilon",
                                    &params.throughput_nn.optimizer_options.epsilon,
                                    0.0f,
                                    0.1f);

                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "relative_decay",
                                    &params.throughput_nn.optimizer_options.relative_decay,
                                    0.0f,
                                    1.0f);

                    throughput_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "absolute_decay",
                                    &params.throughput_nn.optimizer_options.absolute_decay,
                                    0.0f,
                                    1.0f);

                    throughput_model_changed |=
                        IMGUI_STATE1(ImGui::Checkbox,
                                    "adabound",
                                    &params.throughput_nn.optimizer_options.adabound);
                }

                last_active = valid_combo_index(params.throughput_nn.network, network_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Network",
                                      network_types,
                                      network_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(network_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, network_types[i], i == last_active)) {
                            params.throughput_nn.network = (NetworkType)i;
                            throughput_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, network_types);
                }

                last_active = valid_combo_index(params.throughput_nn.loss, loss_types);
                if (IMGUI_STATE_BEGIN(
                        ImGui::BeginCombo, "Loss", loss_types, loss_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(loss_types); ++i)
                        if (IMGUI_STATE(ImGui::Selectable, loss_types[i], i == last_active)) {
                            params.throughput_nn.loss = (LossType)i;
                            throughput_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, loss_types);
                }

                last_active =
                    valid_combo_index(params.throughput_nn.activation, activation_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Activation",
                                      activation_types,
                                      activation_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(activation_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, activation_types[i], i == last_active)) {
                            params.throughput_nn.activation = (ActivationType)i;
                            throughput_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, activation_types);
                }

                last_active = valid_combo_index(params.throughput_nn.output_activation,
                                                activation_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Output Activation",
                                      activation_types,
                                      activation_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(activation_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, activation_types[i], i == last_active)) {
                            params.throughput_nn.output_activation = (ActivationType)i;
                            throughput_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, activation_types);
                }

                IMGUI_STATE(
                    ImGui::InputInt, "n neurons", &params.throughput_nn.n_neurons, 1, 512);

                IMGUI_STATE(ImGui::InputInt,
                            "n hidden layers",
                            &params.throughput_nn.n_hidden_layers,
                            1,
                            512);

                IMGUI_SPACE_SEPARATOR();

                throughput_model_changed |=
                    encoding_ui("Voxel",
                                "Throughput",
                                params.throughput_nn.voxel_encoding,
                                params.throughput_nn.voxel_grid_options);
                IMGUI_SPACE_SEPARATOR();

                throughput_model_changed |=
                    encoding_ui("Outgoing Direction",
                                "Throughput",
                                params.throughput_nn.outgoing_encoding,
                                params.throughput_nn.outgoing_grid_options,
                                &params.throughput_nn.outgoing_sh_options);
                IMGUI_SPACE_SEPARATOR();

                throughput_model_changed |=
                    encoding_ui("Incident Direction",
                                "Throughput",
                                params.throughput_nn.incident_encoding,
                                params.throughput_nn.incident_grid_options,
                                &params.throughput_nn.incident_sh_options);
            }

            IMGUI_STATE_END(ImGui::EndTabItem, &params.throughput_nn);
        }

        if (IMGUI_STATE_BEGIN(
                ImGui::BeginTabItem, "Visibility Optim", &params.visibility_nn)) {
            if (lod_grids_generated) {
                if (params.optimisation_type == OptimisationType::VisibilityOptim &&
                    params.run_optimisation) {
                    auto str = string_format(
                        "Optimising visibility for max resolution %u and %u LoD levels "
                        "(lowest res is %u)",
                        VOXEL_GRID_MAX_RES(),
                        NUM_LODS(),
                        VOXEL_GRID_MIN_RES());
                    IMGUI_VOLATILE(ImGui::Text("%s", str.c_str()));
                }

                IMGUI_STATE1(ImGui::Checkbox,
                            "Compute Inference during training",
                            &params.compute_inference_during_training);

                if (IMGUI_STATE(ImGui::SliderInt,
                                "visibility batch size",
                                (int *)&params.visibility_nn.batch_size,
                                tcnn::batch_size_granularity,
                                DEFAULT_RAY_QUERY_BUDGET,
                                "%d",
                                ImGuiSliderFlags_Logarithmic)) {
                    params.visibility_nn.batch_size =
                        std::min(tcnn::next_multiple(params.visibility_nn.batch_size,
                                                     tcnn::batch_size_granularity),
                                 (uint32_t)DEFAULT_RAY_QUERY_BUDGET);
                }

                IMGUI_SPACE_SEPARATOR();

                IMGUI_STATE(ImGui::SliderFloat,
                            "pdf strength",
                            &params.visibility_nn.pdf_strength,
                            0.f,
                            1000.f);

                IMGUI_STATE(ImGui::SliderFloat,
                            "pdf shift",
                            &params.visibility_nn.pdf_shift,
                            -0.5f,
                            0.5f);

                IMGUI_SPACE_SEPARATOR();

                IMGUI_STATE(ImGui::InputInt,
                            "visibility max training steps",
                            (int *)&params.visibility_nn.n_training_steps);

                if (IMGUI_STATE1(ImGui::Checkbox,
                                "Increase density at boundaries",
                                &params.visibility_nn.increase_boundary_density)) {
                    visibility_model_changed = true;
                    recompute_inference = true;
                }

                if (IMGUI_STATE(ImGui::InputFloat,
                                "Outward voxel extent dilation",
                                &params.visibility_nn.voxel_extent_dilation_outward,
                                0.0f,
                                0.0f,
                                "%.5f")) {
                    params.visibility_nn.voxel_extent_dilation_outward = std::min(
                        std::max(params.visibility_nn.voxel_extent_dilation_outward, 0.0f),
                        1.f);
                    visibility_model_changed = true;
                    recompute_inference = true;
                }

                if (IMGUI_STATE(ImGui::InputFloat,
                                "Inward voxel extent dilation",
                                &params.visibility_nn.voxel_extent_dilation_inward,
                                0.0f,
                                0.0f,
                                "%.5f")) {
                    params.visibility_nn.voxel_extent_dilation_inward = std::min(
                        std::max(params.visibility_nn.voxel_extent_dilation_inward, 0.0f),
                        1.f);
                    visibility_model_changed = true;
                    recompute_inference = true;
                }

                if (IMGUI_STATE(ImGui::InputFloat,
                                "Min voxel dilation",
                                &params.visibility_nn.voxel_extent_min_dilation,
                                0.0f,
                                0.0f,
                                "%.5f")) {
                    params.visibility_nn.voxel_extent_min_dilation =
                        std::min(std::max(params.visibility_nn.voxel_extent_min_dilation, 0.f),
                                 params.visibility_nn.voxel_extent_dilation_outward);
                    visibility_model_changed = true;
                    recompute_inference = true;
                }

                if (IMGUI_STATE(ImGui::InputFloat,
                                "Voxel bound check bias",
                                &params.visibility_nn.voxel_bound_bias,
                                0.0f,
                                0.0f,
                                "%.5f")) {
                    params.visibility_nn.voxel_bound_bias =
                        std::min(std::max(params.visibility_nn.voxel_bound_bias, 0.f), 1.f);
                    visibility_model_changed = true;
                    recompute_inference = true;
                }

                if (IMGUI_STATE_ACTION(ImGui::Button, "Restart Visibility Optimization")) {
                    params.run_optimisation = true;
                    params.optimisation_type = OptimisationType::VisibilityOptim;
                    visibility_model_changed = true;
                    params.compute_optimisation_ref = false;
                }

                if (params.optimisation_type == OptimisationType::VisibilityOptim &&
                    params.run_optimisation &&
                    IMGUI_STATE_ACTION(ImGui::Button, "Pause Visibility Optimization")) {
                    params.run_optimisation = false;
                } else if (params.optimisation_type == OptimisationType::VisibilityOptim &&
                           !params.run_optimisation &&
                           IMGUI_STATE_ACTION(ImGui::Button,
                                              "Continue Visibility Optimization")) {
                    params.run_optimisation = true;
                    params.compute_optimisation_ref = false;
                }

                IMGUI_STATE1(ImGui::Checkbox, "learn single voxel", &params.learn_single_voxel);
            }

            if (IMGUI_STATE_BEGIN_HEADER(ImGui::CollapsingHeader,
                                         "Visibility Neural Net Options",
                                         &params.visibility_nn,
                                         ImGuiTreeNodeFlags_DefaultOpen)) {
                visibility_model_changed |=
                    IMGUI_STATE(ImGui::InputInt,
                                "decay start##Visibility",
                                &params.visibility_nn.optimizer_options.decay_start,
                                0,
                                20000);

                visibility_model_changed |=
                    IMGUI_STATE(ImGui::InputInt,
                                "decay interval##Visibility",
                                &params.visibility_nn.optimizer_options.decay_interval,
                                1,
                                10000);

                visibility_model_changed |=
                    IMGUI_STATE(ImGui::InputFloat,
                                "decay base##Visibility",
                                &params.visibility_nn.optimizer_options.decay_base,
                                0.0f,
                                1.0f);

                int last_active =
                    valid_combo_index(params.visibility_nn.optimizer, optimizer_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Optimizer##Visibility",
                                      optimizer_types,
                                      optimizer_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(optimizer_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, optimizer_types[i], i == last_active)) {
                            params.visibility_nn.optimizer = (OptimizerType)i;
                            visibility_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, optimizer_types);
                }
                if (IMGUI_STATE_BEGIN_HEADER(ImGui::CollapsingHeader,
                                             "Optimizer Options##Visibility",
                                             &params.visibility_nn.optimizer_options,
                                             ImGuiTreeNodeFlags_DefaultOpen)) {
                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "learning_rate##Visibility",
                                    &params.visibility_nn.optimizer_options.learning_rate,
                                    0.0f,
                                    1.0f);

                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "beta1##Visibility",
                                    &params.visibility_nn.optimizer_options.beta1,
                                    0.0f,
                                    1.0f);

                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "beta2##Visibility",
                                    &params.visibility_nn.optimizer_options.beta2,
                                    0.0f,
                                    1.0f);

                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "epsilon##Visibility",
                                    &params.visibility_nn.optimizer_options.epsilon,
                                    0.0f,
                                    0.1f);

                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "relative_decay##Visibility",
                                    &params.visibility_nn.optimizer_options.relative_decay,
                                    0.0f,
                                    1.0f);

                    visibility_model_changed |=
                        IMGUI_STATE(ImGui::InputFloat,
                                    "absolute_decay##Visibility",
                                    &params.visibility_nn.optimizer_options.absolute_decay,
                                    0.0f,
                                    1.0f);

                    visibility_model_changed |=
                        IMGUI_STATE1(ImGui::Checkbox,
                                    "adabound##Visibility",
                                    &params.visibility_nn.optimizer_options.adabound);
                }

                last_active = valid_combo_index(params.visibility_nn.network, network_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Network##Visibility",
                                      network_types,
                                      network_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(network_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, network_types[i], i == last_active)) {
                            params.visibility_nn.network = (NetworkType)i;
                            visibility_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, network_types);
                }

                last_active = valid_combo_index(params.visibility_nn.loss, loss_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Loss##Visibility",
                                      loss_types,
                                      loss_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(loss_types); ++i)
                        if (IMGUI_STATE(ImGui::Selectable, loss_types[i], i == last_active)) {
                            params.visibility_nn.loss = (LossType)i;
                            visibility_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, loss_types);
                }

                last_active =
                    valid_combo_index(params.visibility_nn.activation, activation_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Activation##Visibility",
                                      activation_types,
                                      activation_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(activation_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, activation_types[i], i == last_active)) {
                            params.visibility_nn.activation = (ActivationType)i;
                            visibility_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, activation_types);
                }

                last_active = valid_combo_index(params.visibility_nn.output_activation,
                                                activation_types);
                if (IMGUI_STATE_BEGIN(ImGui::BeginCombo,
                                      "Output Activation##Visibility",
                                      activation_types,
                                      activation_types[last_active])) {
                    for (int i = 0; i < ARRAYSIZE(activation_types); ++i)
                        if (IMGUI_STATE(
                                ImGui::Selectable, activation_types[i], i == last_active)) {
                            params.visibility_nn.output_activation = (ActivationType)i;
                            visibility_model_changed = true;
                        }
                    IMGUI_STATE_END(ImGui::EndCombo, activation_types);
                }

                IMGUI_STATE(ImGui::InputInt,
                            "n neurons##Visibility",
                            &params.visibility_nn.n_neurons,
                            1,
                            512);

                IMGUI_STATE(ImGui::InputInt,
                            "n hidden layers##Visibility",
                            &params.visibility_nn.n_hidden_layers,
                            1,
                            512);

                IMGUI_SPACE_SEPARATOR();

                visibility_model_changed |=
                    encoding_ui("Voxel Joint Entry/Exit",
                                "Visibility",
                                params.visibility_nn.voxel_entry_exit_encoding,
                                params.visibility_nn.voxel_entry_exit_grid_options);
            }

            IMGUI_STATE_END(ImGui::EndTabItem, &params.visibility_nn);
        }

        IMGUI_STATE_END(ImGui::EndTabBar, debug_texture_desc_set);
    }

    if (model_changed) {
        throughput_model_changed = true;
        visibility_model_changed = true;
    }

    renderer_changed |= model_changed;
    return model_changed | throughput_model_changed | visibility_model_changed;
}

void NeuralLodLearning::write_weights_to_disk(const std::string &filename,
                                              NeuralNet *neural_net,
                                              bool serialize_optimizer)
{
    tcnn::json serialized_model = neural_net->trainer->serialize(serialize_optimizer);
    std::ofstream file(filename);
    file << serialized_model;
    file.close();
    println(CLL::INFORMATION, "Serialized weights to : %s", filename.c_str());
}

void NeuralLodLearning::write_config_to_disk(const std::string &filename,
                                             NeuralNet *neural_net)
{
    std::ofstream file(filename);
    file << neural_net->config;
    file.close();
    println(CLL::INFORMATION, "Wrote network config to : %s", filename.c_str());
}

void NeuralLodLearning::load_weights_from_disk(const std::string &filename,
                                               NeuralNet *neural_net)
{
    println(CLL::INFORMATION, "Deserializing weights from : %s", filename.c_str());
    try {
        std::ifstream file(filename);
        neural_net->trainer->deserialize(tcnn::json::parse(file));
        println(CLL::INFORMATION, "Done!", filename.c_str());
    } catch (const std::runtime_error &err) {
        println(CLL::WARNING, "%s", err.what());
    }
}

NEURAL_LOD_LEARNING_NAMESPACE_END