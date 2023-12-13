#!/usr/bin/env python3
## Copyright 2022 Intel Corporation

import sys
import json
import os
import pyvkr
import re
from multiprocessing import Pool

VK_FORMAT_BC1_RGB_UNORM_BLOCK = 131
VK_FORMAT_BC1_RGB_SRGB_BLOCK = 132
VK_FORMAT_BC1_RGBA_UNORM_BLOCK = 133
VK_FORMAT_BC1_RGBA_SRGB_BLOCK = 134
VK_FORMAT_BC3_UNORM_BLOCK = 137
VK_FORMAT_BC3_SRGB_BLOCK = 138
VK_FORMAT_BC5_UNORM_BLOCK = 141

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} SCENE_FILE TEXTURE_SRC_DIR")
    sys.exit(1)

sceneFile = sys.argv[1]
overrideFile = sceneFile + ".json"

overrides = {}
try:
    with open(overrideFile) as file:
        ovr = json.load(file)
        if "replace" in ovr:
            overrides = ovr["replace"]
except FileNotFoundError:
    print(f"To define texture overrides, use {overrideFile}.")

scriptDir = os.path.dirname(os.path.realpath(__file__))
srcTextureDir = sys.argv[2]
scene = pyvkr.open_scene(sceneFile)
textureDir = scene['textureDir']
os.makedirs(textureDir, exist_ok=True)
materials = scene['materials']
numMaterials = len(materials)

def check_src(tex, default, fmt, ofmt=None):
    srcFile = f"{tex}.png"

    for rule in overrides:
        for (pat, sub) in rule.items():
            srcFile = re.sub(pat, sub, srcFile)

    defaultSrc = os.path.join(scriptDir, 'default_textures', default)

    if not srcFile:
        src = defaultSrc
    else:
        src = os.path.join(srcTextureDir, srcFile)

    tgtFile = f"{tex}.vkt"
    tgt = os.path.join(textureDir, tgtFile)

    if ofmt is None:
        ofmt = fmt
    if os.path.exists(src):
        return (src, tgt, (fmt, ofmt), False, srcFile)

    return (defaultSrc, tgt, (fmt, ofmt), True, srcFile)


def convert_basecolor(material):
    src = f"{material}_BaseColor"
    return check_src(src, "zero.png", VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_BC1_RGB_SRGB_BLOCK)

def convert_specular(material):
    src = f"{material}_Specular"
    return check_src(src, "green.png", VK_FORMAT_BC1_RGB_UNORM_BLOCK)

def convert_normal(material):
    src = f"{material}_Normal"
    return check_src(src, "zero_normal.png", VK_FORMAT_BC5_UNORM_BLOCK)

def cb(e):
    print(f"multiprocessing error: {e}")
    sys.exit(1)

with Pool() as pool:
    textures = []
    for i, mat in enumerate(materials):
        material = mat['name']
        textures.append(convert_basecolor(material))
        textures.append(convert_specular(material))
        textures.append(convert_normal(material))

    missingTextures = [ t[4] for t in textures if t[3] ]

    if missingTextures:
        missingTextures.sort()
        print("The following source textures could not be found and will be converted from default textures.")
        print(f"""You may paste this snippet into {overrideFile} and provide overrides.
Replace with an empty string to silence the warning but still use a default texture.""")
        print("{")
        print('  "replace": [')
        print(",\n".join(f'    {{ "{t}": "" }}' for t in missingTextures))
        print("  ]")
        print("}")

    results = [ (pool.apply_async(pyvkr.convert_texture, (t[0], t[1], *(t[2] if isinstance(t[2], tuple) else (t[2],)))), t)
        for t in textures ]

    numAsync = len(results)
    failed = []
    for i, (r, t) in enumerate(results):
        sys.stdout.write(f"{i} / {numAsync} converted ...\r")
        sys.stdout.flush()
        r.wait()
        if not r.successful():
            msg = ""
            try:
                result = r.get()
            except Exception as exp:
                msg = exp
            failed.append((t, msg))

    print(f"{numAsync} / {numAsync} converted ...")

    if failed:
        print('Conversion errors for the following files:')
    for t, msg in failed:
        print(f"{t[0]}: {msg}")


