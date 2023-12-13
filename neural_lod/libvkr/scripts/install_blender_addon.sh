#!/bin/sh
# Copyright 2022 Intel Corporation.

SCRIPT_FILE=$(readlink -f ${0})
SCRIPT_DIR=$(dirname ${SCRIPT_FILE})
MODULE_ARCHIVE="${SCRIPT_DIR}/blender_vkr.zip"

if ! ( command -v blender &> /dev/null )
then
  echo "Cannot find blender"
  exit 1
fi

if [ ! -f "${MODULE_ARCHIVE}" ]
then
  echo "Cannot find module archive ${MODULE_ARCHIVE}"
  exit 2
fi

INSTALL_SCRIPT='
import bpy
import os
import sys

module_name = "blender_vkr"
script_dir = os.getcwd()
module_file = os.path.join(script_dir, module_name + ".zip")

if not os.path.isfile(module_file):
  print(f"Cannot find module archive {module_file}")
  sys.exit(1)

bpy.ops.preferences.addon_install(overwrite=True, filepath=module_file)
bpy.ops.preferences.addon_enable(module=module_name)
# Required so that blender does not forget we enabled.
bpy.ops.wm.save_userpref()
'

cd "${SCRIPT_DIR}"
blender --background --python-expr "${INSTALL_SCRIPT}"
