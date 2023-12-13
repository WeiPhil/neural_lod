#include "wavefront_neural_ref.h"
#include "../../imstate.h"
#include "scene.h"

void WavefrontConstantRef::release_mapped_display_resources()
{
    this->WavefrontPT::release_mapped_display_resources();
}

bool WavefrontConstantRef::ui_and_state(bool &renderer_changed)
{
    bool scene_changed = this->WavefrontPT::ui_and_state(renderer_changed);

    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "use envmap", &params.use_envmap);

    if (params.use_envmap) {
        IMGUI_STATE_(
            ImGui::InputText, "envmap filename", params.envmap_filename, MAX_FILENAME_SIZE);
        if (IMGUI_STATE_ACTION(ImGui::Button, "reload envmap")) {
            load_envmap();
            renderer_changed = true;
        }
        scene_changed |= IMGUI_STATE(
            ImGui::DragFloat, "envmap rotation", &params.envmap_rotation, 0.5, 0.f, 360.f);
    } else {
        scene_changed |= IMGUI_STATE3(ImGui::ColorEdit,
                                      "envmap color",
                                      &params.envmap_color[0],
                                      ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        scene_changed |= IMGUI_STATE3(ImGui::ColorEdit,
                                      "background color",
                                      &params.background_color[0],
                                      ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
    }

    renderer_changed |= IMGUI_STATE(ImGui::InputInt, "max depth", &params.max_depth);

    renderer_changed |=
        IMGUI_STATE1(ImGui::Checkbox, "apply russian roulette", &params.apply_rr);

    renderer_changed |=
        IMGUI_STATE1(ImGui::Checkbox, "show visibility map", &params.show_visibility_map);

    return scene_changed | renderer_changed;
}

std::string WavefrontConstantRef::name() const
{
    return "Cuda Neural Radiance Caching Extension";
}

void WavefrontConstantRef::update_scene_from_backend(const Scene &scene)
{
    this->WavefrontPT::update_scene_from_backend(scene);

    scene_aabb = {.min = glm::vec3(2e32f), .max = glm::vec3(-2e32f)};

    for (auto const &instance : scene.instances) {
        auto &parameterized_mesh = scene.parameterized_meshes[instance.parameterized_mesh_id];
        auto &mesh = scene.meshes[parameterized_mesh.mesh_id];

        const auto &animData = scene.animation_data.at(instance.animation_data_index);
        constexpr uint32_t frame = 0;
        const glm::mat4 instance_transform =
            animData.dequantize(instance.transform_index, frame);

        println(CLL::VERBOSE,
                "[WavefrontConstantRef] Computing scene bbox and voxels bbox",
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

    // This is coherent with the voxelisation done in the cuda-voxelize library
    voxel_grid_aabb = {.min = scene_aabb.min, .max = scene_aabb.max};
    glm::vec3 lengths =
        scene_aabb.max - scene_aabb.min;  // check length of given bbox in every direction
    float max_length = glm::max(lengths.x, glm::max(lengths.y, lengths.z));  // find max length
    for (unsigned int i = 0; i < 3; i++) {  // for every direction (X,Y,Z)
        if (max_length == lengths[i]) {
            continue;
        } else {
            float delta = max_length - lengths[i];  // compute difference between largest
                                                    // length and current (X,Y or Z) length
            voxel_grid_aabb.min[i] =
                scene_aabb.min[i] -
                (delta / 2.0f);  // pad with half the difference before current min
            voxel_grid_aabb.max[i] =
                scene_aabb.max[i] +
                (delta / 2.0f);  // pad with half the difference behind current max
        }
    }
}
