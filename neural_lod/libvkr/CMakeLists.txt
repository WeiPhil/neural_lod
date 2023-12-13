## Copyright 2022 Intel Corporation

option(LIBVKR_ENABLE_PYTHON "Enable module for using libvkr in Python" ON)

if (LIBVKR_ENABLE_PYTHON)
  cmake_minimum_required(VERSION 3.18.4) # introduces Development.Module
else ()
  cmake_minimum_required(VERSION 3.16.3)
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(libvkr VERSION 1.0 LANGUAGES C)

# Uncomment to compile with address sanitizer.
# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_C_COMPILER clang)

add_library(vkr STATIC src/vkr.c)
target_include_directories(vkr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(vktconvert src/vktconvert.c)
target_link_libraries(vktconvert PRIVATE vkr)

# On some systems, we need to link against libm to use pow.
# Only do that if libm exists, though.
include(CheckLibraryExists)
check_library_exists(m pow "" NEED_LIBM)
if (NEED_LIBM)
  target_link_libraries(vktconvert PRIVATE m)
endif()

include(cmake/blender_addon.cmake)

# The python module is an optional component, but we require it
# for our conversion utilities.
if (LIBVKR_ENABLE_PYTHON)

  detect_blender()

  if (HAVE_BLENDER)
    # Infer python versions from Blender if we have it.
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
      set(AUTO_PYTHON_VERSION "3...<${BLENDER_PYTHON_VERSION_NEXT}")
    else()
      set(AUTO_PYTHON_VERSION "${BLENDER_PYTHON_VERSION}")
    endif()
    set(MAX_NUMPY_VERSION "${BLENDER_NUMPY_VERSION}")
  else()
    # Defaults that are compatible with more than just the latest Blender.
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
      set(AUTO_PYTHON_VERSION "3...<3.10")
    else()
      set(AUTO_PYTHON_VERSION "3.9")
    endif()
    set(MAX_NUMPY_VERSION "1.21")
  endif()

  # Provide a user override.
  set(LIBVKR_PYTHON_VERSION "${AUTO_PYTHON_VERSION}" CACHE STRING
      "Select python version (append EXACT if necessary")

  find_package(
    Python ${LIBVKR_PYTHON_VERSION}
    COMPONENTS Development.Module NumPy
  )

  if (Python_FOUND AND TARGET Python::Module AND TARGET Python::NumPy)
    message(STATUS "Found Python ${Python_VERSION} and NumPy ${Python_NumPy_VERSION}."
      " Will build pyvkr.")
    Python_add_library(pyvkr MODULE src/vkrmodule.c)
    target_link_libraries(pyvkr PRIVATE vkr Python::NumPy)

    if (Python_NumPy_VERSION VERSION_LESS_EQUAL MAX_NUMPY_VERSION)
      blender_addon_copy_target(pyvkr)
    else()
      message(WARNING "For blender support, the NumPy version must be less than ${MAX_NUMPY_VERSION}."
        " The blender addon will not support pyvkr.")
    endif()
  else()
    message(STATUS "Not building pyvkr, could not find Python/Numpy-dev (${Python_VERSION}/${Python_NumPy_VERSION})")
  endif()
endif ()

blender_addon_build(
  __init__.py
  operator_file_export_vkrs.py
)
