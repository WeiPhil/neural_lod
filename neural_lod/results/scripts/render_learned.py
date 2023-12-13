import os
import sys
from subprocess import PIPE, run
import argparse

from pyrsistent import optional

os.chdir("../../../rptr/build")

parser = argparse.ArgumentParser(description='Lod Generator')
parser.add_argument("scene_path", help="scene filename path without any suffix or extension (relative from CRT build dir or absolute path)")
parser.add_argument('--config_path', help='scene config file pointing to learned weights (relative from CRT build dir or absolute path)', required=True)
parser.add_argument('--result_dir', help='directory path for results (relative from CRT build dir or absolute path)', required=True)
parser.add_argument('--num_lods', type=int, default=7, help='number of processed lods (default: 7)')
parser.add_argument('--max_pow_2', type=int, default=9, help='the highest resolution power of 2 (default: 9 -> 2^9 = 512)')
parser.add_argument('--spp', type=int, default=512, help='samples per pixel for lods (default: 512)')
parser.add_argument('--ref_spp', type=int, default=512, help='samples per pixel for reference (default: 512)')
parser.add_argument('--fov', type=float, default=65.0, help='optional fov parameter for CRT (default: 65.0)')
parser.add_argument('--up', type=float, nargs=3, default=[0.0,1.0,0.0], help='optional up camera parameter for CRT (default: 0.0 1.0 0.0)')
parser.add_argument('--thresholds', type=float, nargs='*', default=[0.5], help='vector of thresholds or single trheshold to use for all lods (default: 0.5)')
parser.add_argument('--occupation_fraction', type=float, default=1.0, help='render to image occupation fraction for lod comparisons (default: 1.0)')
parser.add_argument('--skip_ref', action='store_true')
parser.add_argument('--skip_present', action='store_true')
parser.add_argument('--res', type=int, default=512, help='squared resolution of the rendered image')
parser.add_argument('--no_vis_rr', action='store_true')
parser.add_argument('--no_auto_vis_rr', action='store_true')
 
args = parser.parse_args()

if not os.path.exists(os.path.dirname(args.result_dir)):
        os.makedirs(os.path.dirname(args.result_dir))

thresholds_values = None

if(len(args.thresholds) == 1):
    thresholds_values = [args.thresholds[0] for _ in range(0,args.num_lods)]
elif(len(args.thresholds) == args.num_lods):
    thresholds_values = args.thresholds
else:
    sys.exit('The length of --thresholds should either match the number of lods or not specified to default to 0.5')

def run_cmd(command : str,ignore_program_output = False):
    print("Running : {}".format(command))
    stdout_type = PIPE if ignore_program_output else None 
    stderr_type = PIPE if ignore_program_output else None 
    result = run(command,stdout=stdout_type, stderr=stderr_type, universal_newlines=True, shell=True)
    if result.stderr != None:
        print(result.stderr)

def create_tmp_config(config_file : str, is_ref : bool = False, lod : int = 0, threshold = 0.5):
    data = None
    with open(config_file, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    res = max_res >> lod

    for i in range(0,len(data)):
        # print(data[i])
        if(data[i].startswith("[.][*variant]")):
            data[i+1] = "Wavefront Neural Ref= 1\n" if is_ref else "Wavefront Neural Throughput Visibility Lod= 1\n"
        if(data[i].startswith("[.][current lod]")):
            data[i+1] = "LoD {}= 1\n".format(lod)
        # remove weigths loading
        if(is_ref and (data[i].startswith("Weights directory=") or data[i].startswith("Weights filename prefix="))):
            data[i] = ""
        if(data[i].startswith("apply visibility russian roulette=")):
            if(not args.no_auto_vis_rr and not args.no_vis_rr):
                data[i] = "apply visibility russian roulette= 1\n" if res >= 128 else "apply visibility russian roulette= 0\n"
            else:
                data[i] = "apply visibility russian roulette= 0\n" if args.no_vis_rr else "apply visibility russian roulette= 1\n"
        if(data[i].startswith("render wavefront extensions")):
            data[i] = "render wavefront extensions= 1\n"
        if(data[i].startswith("inferred visibility threshold=")):
            data[i] = "inferred visibility threshold= {}\n".format(threshold)
        if(data[i].startswith("max depth=")):
            data[i] = "max depth= 1000\n"
        if(data[i].startswith("rr start bounce= ")):
            data[i] = "rr start bounce= 1\n"
        if(data[i].startswith("auto downsampling= ")):
            data[i] = "auto downsampling= 0\n"
        if(data[i].startswith("window width= ")):
            data[i] = "window width= {}\n".format(args.res)
        if(data[i].startswith("window height= ")):
            data[i] = "window height= {}\n".format(args.res)

    config_file_base_path = os.path.dirname(config_file)
    print("config_file_base_path :",config_file_base_path)
    tmp_filename = config_file_base_path + '/tmp.ini'
    with open(tmp_filename, 'w') as file:
        file.writelines( data )
    return tmp_filename

lod_spp = args.spp
ref_spp = args.ref_spp

max_res = 2**args.max_pow_2

result_base_filename = os.path.splitext(os.path.basename(args.config_path))[0]

crt_base_command = "./rptr {} ".format(args.scene_path)
crt_base_command += "--fov {} ".format(args.fov)
crt_base_command += "--up {} {} {} ".format(args.up[0],args.up[1],args.up[2])
crt_base_command += "--exr "

if not args.skip_ref:
    ## ref file
    ref_config_filename = create_tmp_config(args.config_path, is_ref=True)
    res = max_res
    
    result_filename_prefix_path = args.result_dir + result_base_filename + "_spp_{}_ref".format(lod_spp)

    command = crt_base_command
    command += "--validation {} ".format(result_filename_prefix_path)
    command += "--validation-spp {} ".format(ref_spp)
    command += "--config {} ".format(ref_config_filename)
    run_cmd(command)

    run_cmd("rm {}".format(ref_config_filename))

    result_file = "{}_{}.exr".format(result_filename_prefix_path,str(ref_spp).zfill(4))
    desired_result_file = "{}.exr".format(result_filename_prefix_path)
    command = "mv {} {}".format(result_file,desired_result_file)
    run_cmd(command)

# lod files
for lod in range(0,args.num_lods):
    res = max_res >> lod
    print("Processing lod resolution {} at {} spp".format(res,lod_spp))
    lod_config_filename = create_tmp_config(args.config_path, is_ref=False, lod=lod, threshold= thresholds_values[lod])
    result_filename_prefix_path = args.result_dir + result_base_filename + "_spp_{}_lod_{}".format(lod_spp,res)
    
    desired_result_file_exr = "{}.exr".format(result_filename_prefix_path)
    # desired_result_file_png = "{}.png".format(result_filename_prefix_path)
    if(args.skip_present and os.path.exists(desired_result_file_exr)):
        print("Skipping already present file {}".format(desired_result_file_exr))
        continue

    command = crt_base_command
    command += "--validation {} ".format(result_filename_prefix_path)
    command += "--validation-spp {} ".format(lod_spp)
    command += "--config {} ".format(lod_config_filename)
    run_cmd(command)

    run_cmd("rm {}".format(lod_config_filename))

    result_file_exr = "{}_{}.exr".format(result_filename_prefix_path,str(lod_spp).zfill(4))
    command = "mv {} {}".format(result_file_exr,desired_result_file_exr)
    run_cmd(command)

result_filename_prefix = args.result_dir + result_base_filename + "_spp_{}".format(lod_spp)

thr_string = "_thr_"
if(len(args.thresholds) == 1):
    thr_string += "{:.3f}".format(thresholds_values[0])
else:
    for thr in thresholds_values:
        thr_string += "{:.3f}_".format(thr)
    thr_string = thr_string[:-1]

command = "python3 ../../neural_lod/results/scripts/lod_comparison_fig.py {} --skip_out --num_lods {} --max_res {} --angle 40 --occupation_fraction {} --postfix_string {}".format(
    result_filename_prefix, args.num_lods, max_res, args.occupation_fraction, thr_string)
run_cmd(command)
