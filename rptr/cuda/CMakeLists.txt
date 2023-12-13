option(ENABLE_NEURAL_LOD "Enable neural rendering techniques" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

if (ENABLE_NEURAL_LOD)
    set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "disable benchmarks")
    set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "disable examples")
    set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "default CUDA architecture" FORCE)
    add_subdirectory(../ext/tiny-cuda-nn tiny-cuda-nn EXCLUDE_FROM_ALL)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Xcudafe --diag_suppress=20012") # diag: defaulted host/device functions may differ. to update: -Xcudafe --display_error_number
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_USE_MATH_DEFINES")
    set(NEURAL_COMPILE_DEFINITIONS ${TCNN_DEFINITIONS} -DENABLE_NEURAL_LOD)
    set(NEURAL_INCLUDE_PATHS ../ext/tiny-cuda-nn/include ../ext/tiny-cuda-nn/dependencies)
    set(NEURAL_LIBRARIES tiny-cuda-nn)

    set(NEURAL_SOURCES
        neural-lod/neural_lod_learning_init.cu
        neural-lod/neural_lod_learning.cu
        neural-lod/neural_lod_learning_ui.cu
        )
endif ()

add_library(render_cuda
    render_cuda.cpp
    wavefront_neural_pt.cu
    wavefront_pt.cu
    wavefront_pt.cpp
    neural-lod/wavefront_neural_ref.cu
    neural-lod/wavefront_neural_ref.cpp 
    neural-lod/wavefront_neural_visibility_lod.cu
    neural-lod/wavefront_neural_visibility_lod_ui.cu 
    neural-lod/wavefront_neural_throughput_visibility_lod.cu
    neural-lod/wavefront_neural_throughput_visibility_lod_ui.cu

    ${NEURAL_SOURCES}
)
target_precompile_headers(render_cuda PRIVATE pch.hpp)

target_compile_definitions(render_cuda PUBLIC ${NEURAL_COMPILE_DEFINITIONS})
target_include_directories(render_cuda PUBLIC ${NEURAL_INCLUDE_PATHS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Inlining everything for now, as external linking seems broken w/ constant memory
#set_target_properties(render_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(render_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#target_compile_options(render_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code true>)
#target_link_options(render_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code true>)

target_link_libraries(render_cuda PUBLIC
	librender util ${NEURAL_LIBRARIES}) # ${CUDA_LIBRARIES}
