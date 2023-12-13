#include "wavefront_neural_throughput_visibility_lod.h"
#include "../../imstate.h"

bool WavefrontNeuralThroughputVisibilityLod::ui_and_state() {
    bool renderer_changed = this->WavefrontNeuralPT::ui_and_state();

    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "use envmap", &params.use_envmap);

    if(params.use_envmap){
        IMGUI_STATE_(ImGui::InputText, "envmap filename", params.envmap_filename, MAX_FILENAME_SIZE);
        if(IMGUI_STATE_ACTION(ImGui::Button, "reload envmap")){
            load_envmap();
            renderer_changed = true;
        }
        renderer_changed |= IMGUI_STATE(ImGui::DragFloat, "envmap rotation", &params.envmap_rotation, 0.5, 0.f,360.f);
    }else{
        renderer_changed |= IMGUI_STATE3(ImGui::ColorEdit, "envmap color", &params.envmap_color[0], ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        renderer_changed |= IMGUI_STATE3(ImGui::ColorEdit, "background color", &params.background_color[0], ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
    }

    renderer_changed |= IMGUI_STATE(ImGui::InputInt, "max depth", (int *)&params.max_depth);
    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "apply russian roulette", &params.apply_rr);
    if(params.apply_rr){
        renderer_changed |= IMGUI_STATE(ImGui::InputInt, "rr start bounce", (int *)&params.rr_start_bounce);
    }
    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "apply visibility russian roulette", &params.apply_visibility_rr);

    IMGUI_SPACE_SEPARATOR();
    
    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "use henyey greenstein sampling", &params.use_hg);
    renderer_changed |= IMGUI_STATE(ImGui::SliderFloat, "henyey greenstein g", &params.hg_g, -0.99999f, 0.99999f);
    
    IMGUI_SPACE_SEPARATOR();

    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "stochastic threshold", &params.stochastic_threshold);

    return renderer_changed;
}