#include "wavefront_neural_visibility_lod.h"
#include "../../imstate.h"

bool WavefrontNeuralVisibilityLod::ui_and_state() {
    bool renderer_changed = this->WavefrontNeuralPT::ui_and_state();

    renderer_changed |= IMGUI_STATE3(ImGui::ColorEdit, "visibility color", &params.visibility_color[0], ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
    renderer_changed |= IMGUI_STATE3(ImGui::ColorEdit, "background color", &params.background_color[0], ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
    renderer_changed |= IMGUI_STATE(ImGui::InputInt, "max visibility inferences", (int*)&params.max_visibility_inferences);

    params.max_visibility_inferences = max(params.max_visibility_inferences,1);
    
    IMGUI_SPACE_SEPARATOR();

    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "stochastic threshold", &params.stochastic_threshold);
    
    IMGUI_SPACE_SEPARATOR();

    renderer_changed |= IMGUI_STATE1(ImGui::Checkbox, "[DEBUG] display needed inferences", &params.display_needed_inferences);

    return renderer_changed;
}
