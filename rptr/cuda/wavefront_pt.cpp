#include "wavefront_pt.h"
#include "../../imstate.h"

WavefrontPT::WavefrontPT(RenderCudaBinding* backend)
    : backend(backend) {
    rt_rayquery_index = backend->backend->variant_index("RQ_CLOSEST");
}

WavefrontPT::~WavefrontPT() {
}

void WavefrontPT::release_mapped_display_resources() {
    checkCuda(cudaFree(this->throughtput_pdf));
    this->throughtput_pdf = nullptr;
    checkCuda(cudaFree(this->illumination_rnd));
    this->illumination_rnd = nullptr;
    checkCuda(cudaFree(this->nee_pdf));
    this->nee_pdf = nullptr;
}

void WavefrontPT::initialize(const int fb_width, const int fb_height) {
    this->WavefrontPT::release_mapped_display_resources();

    checkCuda(cudaMalloc((void**) &this->throughtput_pdf, sizeof(float) * 4 * fb_width * fb_height));
    checkCuda(cudaMalloc((void**) &this->illumination_rnd, sizeof(float) * 4 * fb_width * fb_height));
    checkCuda(cudaMalloc((void**) &this->nee_pdf, sizeof(float) * 4 * fb_width * fb_height));
}

bool WavefrontPT::ui_and_state(bool& renderer_changed) {
    return false;
}

std::string WavefrontPT::name() const {
    return "Cuda Wavefront Path Tracing Extension";
}

void WavefrontPT::update_scene_from_backend(const Scene& scene) {
    
}
