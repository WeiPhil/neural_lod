#!/bin/sh
# Copyright 2022 Intel Corporation.

import bpy

# NOTE: If you use pyvkr functionality, please guard that by checking
#       have_pyvkr.
try:
    from . import pyvkr
    have_pyvkr = True
except:
    print(f"Failed to load pyvkr. This probably means that your blender version is quite old; try a more recent one!")
    have_pyvkr = False

bl_info = {
    "name": ".vks/.vkt [blender_vkr]",
    "blender": (3,1,0),
    "category": "Export",
}

from . import operator_file_export_vkrs as vks

def register():
    bpy.utils.register_class(vks.ExportVulkanRendererScene)
    bpy.types.TOPBAR_MT_file_export.append(vks.menu_func_export)

def unregister():
    bpy.utils.unregister_class(vks.ExportVulkanRendererScene)
    bpy.types.TOPBAR_MT_file_export.remove(vks.menu_func_export)

# Test call, blender does not execute this.
if __name__ == "__main__":
    register()
    bpy.ops.export.vulkan_renderer_scene('EXEC_DEFAULT',
        filepath="export-test-run2.vks")
