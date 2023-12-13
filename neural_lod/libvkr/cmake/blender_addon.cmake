#
# Copyright 2022 Intel Corportation.
#

set(BLENDER_ADDON blender_vkr)
set(BLENDER_ADDON_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${BLENDER_ADDON})
set(BLENDER_ADDON_ARCHIVE ${CMAKE_BINARY_DIR}/${BLENDER_ADDON}.zip)
set(BLENDER_ADDON_SOURCE ${PROJECT_SOURCE_DIR}/scripts/${BLENDER_ADDON})

# This function attempts to find the Blender binary, and if it can find one
# detects the Python and NumPy versions used in Blender.
# Sets HAVE_BLENDER. If Blender was found, also sets BLENDER_PYTHON_VERSION
# and BLENDER_NUMPY_VERSION (both contain major and minor only).
# Also sets BLENDER_PYTHON_VERSION_NEXT for range comparisons.
function(detect_blender)
  find_program(HAVE_BLENDER blender)

  if (HAVE_BLENDER)
    set(EXPRESSION "import sys")
    set(EXPRESSION "${EXPRESSION}; import numpy")
    set(EXPRESSION "${EXPRESSION}; vi=sys.version_info")
    set(EXPRESSION "${EXPRESSION}; print(f'python_version={vi.major}.{vi.minor}')")
    set(EXPRESSION "${EXPRESSION}; print(f'numpy_version={numpy.__version__}')")
    execute_process(
      COMMAND ${HAVE_BLENDER} --background --python-expr "${EXPRESSION}"
      OUTPUT_VARIABLE BLENDER_PYTHON_VERSION_INFO
    )
    string(
      REGEX MATCH "python_version=([0-9]+)\.([0-9]+)" 
      BLENDER_PYTHON_VERSION_MATCH 
      "${BLENDER_PYTHON_VERSION_INFO}"
    )
    set(BLENDER_PYTHON_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(BLENDER_PYTHON_VERSION_MINOR "${CMAKE_MATCH_2}")
    math(EXPR BLENDER_PYTHON_VERSION_NEXT_MINOR "${BLENDER_PYTHON_VERSION_MINOR}+1")
    set(BLENDER_PYTHON_VERSION
      "${BLENDER_PYTHON_VERSION_MAJOR}.${BLENDER_PYTHON_VERSION_MINOR}")
    set(BLENDER_PYTHON_VERSION_NEXT 
      "${BLENDER_PYTHON_VERSION_MAJOR}.${BLENDER_PYTHON_VERSION_NEXT_MINOR}")

    string(
      REGEX MATCH "numpy_version=([0-9]+)\.([0-9]+)\.([0-9]+)" 
      BLENDER_NUMPY_VERSION_MATCH 
      "${BLENDER_PYTHON_VERSION_INFO}"
    )
    set(BLENDER_NUMPY_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(BLENDER_NUMPY_VERSION_MINOR "${CMAKE_MATCH_2}")
    math(EXPR BLENDER_NUMPY_VERSION_NEXT_MINOR "${BLENDER_NUMPY_VERSION_MINOR}+1")
    set(BLENDER_NUMPY_VERSION
      "${BLENDER_NUMPY_VERSION_MAJOR}.${BLENDER_NUMPY_VERSION_MINOR}")
    set(BLENDER_NUMPY_VERSION_NEXT 
      "${BLENDER_NUMPY_VERSION_MAJOR}.${BLENDER_NUMPY_VERSION_NEXT_MINOR}")

    set(BLENDER_PYTHON_VERSION_NEXT "${BLENDER_PYTHON_VERSION_NEXT}" PARENT_SCOPE)
    set(BLENDER_PYTHON_VERSION "${BLENDER_PYTHON_VERSION}" PARENT_SCOPE)
    set(BLENDER_NUMPY_VERSION "${BLENDER_NUMPY_VERSION}" PARENT_SCOPE)

    message(STATUS "Found Blender with "
      "Python ${BLENDER_PYTHON_VERSION} and NumPy ${BLENDER_NUMPY_VERSION}: "
      ${HAVE_BLENDER})
  else()
    message(STATUS "Did not find Blender")
  endif()

  set(HAVE_BLENDER "${HAVE_BLENDER}" PARENT_SCOPE)
endfunction()

# This has to be first to set up the archive target.
add_custom_command(OUTPUT ${BLENDER_ADDON_ARCHIVE}
  COMMENT "Building ${BLENDER_ADDON_ARCHIVE} ..."
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${BLENDER_ADDON_OUTPUT_DIR} # backwards-compatible for `rm -rf`
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BLENDER_ADDON_OUTPUT_DIR}
)

# Optionally add target output to the addon. We use this to copy pyvkr.
function(blender_addon_copy_target TGT)
  add_custom_command(
    APPEND OUTPUT ${BLENDER_ADDON_ARCHIVE}
    DEPENDS ${TGT}
    COMMAND ${CMAKE_COMMAND} -E
      copy $<TARGET_FILE:${TGT}> ${BLENDER_ADDON_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TGT}>
  )
endfunction()

# Takes a list of files in scripts/blender_vkr as optional arguments.
function(blender_addon_build)
  foreach(F IN LISTS ARGN)
    set(IN_FILE ${BLENDER_ADDON_SOURCE}/${F})
    set(OUT_FILE ${BLENDER_ADDON_OUTPUT_DIR}/${F})
    get_filename_component(FILE_DIR ${F} DIRECTORY)
    add_custom_command(
      APPEND OUTPUT ${BLENDER_ADDON_ARCHIVE}
      DEPENDS ${IN_FILE}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${BLENDER_ADDON_OUTPUT_DIR}/${FILE_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy ${IN_FILE} ${OUT_FILE}
    )
  endforeach()

  add_custom_command(
    APPEND OUTPUT ${BLENDER_ADDON_ARCHIVE}
    COMMAND ${CMAKE_COMMAND} -E tar c ${BLENDER_ADDON_ARCHIVE} --format=zip
      ${BLENDER_ADDON_OUTPUT_DIR}
  )

  add_custom_target(build_blender_addon ALL
    DEPENDS ${BLENDER_ADDON_ARCHIVE}
  )

  if(UNIX AND HAVE_BLENDER)
    add_custom_target(install_blender_addon
      DEPENDS ${BLENDER_ADDON_ARCHIVE}
      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/scripts/install_blender_addon.sh
        ${CMAKE_BINARY_DIR}
      COMMAND ${CMAKE_BINARY_DIR}/install_blender_addon.sh
    )
  endif()
endfunction()
