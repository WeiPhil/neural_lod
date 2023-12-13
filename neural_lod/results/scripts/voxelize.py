import os
import sys
from subprocess import PIPE, run
import argparse

from pyrsistent import optional

parser = argparse.ArgumentParser(description='Base Scene Voxelisation')
parser.add_argument("input_vks", help="the vks file to process")
parser.add_argument('output', default=512, help='output path file for obj and lod pointcloud')
parser.add_argument('--max_res', default=512, help='max lod resolution')
parser.add_argument('--add_padded', action='store_false', default=True, help='as a second output, slightly pad the aabb to avoid floating point issues')
args = parser.parse_args()

input_filename = args.input_vks
max_res = args.max_res
output_path = args.output
add_padded = args.add_padded

def run_cmd(command : str,ignore_program_output = False):
    print("Running : {}".format(command))
    stdout_type = PIPE if ignore_program_output else None 
    stderr_type = PIPE if ignore_program_output else None 
    result = run(command,stdout=stdout_type, stderr=stderr_type, universal_newlines=True, shell=True)
    if result.stderr != None:
        print(result.stderr)

vkr2obj_path = "../../libvkr/scripts/vkr2obj.py"

obj_output_path = output_path + ".obj"

vkr2obj_cmd = "python3 {} {} > {}".format(vkr2obj_path,input_filename,output_path + ".obj")

run_cmd(vkr2obj_cmd)

cudavoxelizer_cmd = "../../cuda_voxelizer/build/cuda_voxelizer -f {} -s {} -thrust -o obj_points".format(obj_output_path,max_res)

run_cmd(cudavoxelizer_cmd)

if(add_padded):
    cudavoxelizer_cmd += " -pad"
    run_cmd(cudavoxelizer_cmd)