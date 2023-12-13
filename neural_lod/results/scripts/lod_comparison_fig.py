import figuregen
from figuregen.util import image
import simpleimageio as sio
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
import argparse

from flip.flip import compute_flip
from flip.utils import *

def compute_flip_error(img, ref,background_color):

    # Set viewing conditions
    monitor_distance = 0.7
    monitor_width = 0.7
    monitor_resolution_x = 3840

    # Compute number of pixels per degree of visual angle
    pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

    # Swap axes for the flip library
    ref_r = ref[:,:,0]
    ref_g = ref[:,:,1]
    ref_b = ref[:,:,2]

    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    ref = np.stack([ref_r, ref_g, ref_b], axis=0)
    img = np.stack([img_r, img_g, img_b], axis=0)

    # Compute FLIP map
    deltaE = compute_flip(ref,img, pixels_per_degree)

    # Save error map
    index_map = np.floor(255.0 * deltaE.squeeze(0))

    use_color_map = True
    if use_color_map:
        result = CHWtoHWC(index2color(index_map, get_magma_map()))
    else:
        result = index_map / 255.0

    return result


def compute_relMSE_error(img, ref, dynamic_range: int = +6, exposure: int = +11, color_map = "viridis", epsilon: float = 0.0001):
        cmap = plt.get_cmap(color_map)
        relMSE = image.relative_squared_error(img,ref,epsilon).mean(axis=-1)
        relMSE = relMSE * 2**exposure
        relMSE = np.log2(relMSE + 2*(-dynamic_range/2)) / dynamic_range + 0.5
        relMSE = np.clip(relMSE, 0, 1)
        im = cmap(relMSE)
        return im

def proportional_resize(image, width = None, height = None, rescale = 1.0, interpolation = cv2.INTER_NEAREST):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = ( width, int(h * r))

    dim = (int(dim[0] * rescale),int(dim[1] * rescale))
    print(dim)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = interpolation)

    # return the resized image
    return resized

def load_images(file_prefix, resolutions, occupation_fraction = 0.8, hdr_extension = '.exr'):

    ref_image = image.lin_to_srgb(sio.read(file_prefix + "_ref" + hdr_extension))

    img_empty = np.tile([1.0,1.0,1.0], (ref_image.shape[0], ref_image.shape[1], 1))

    scale_x = True if ref_image.shape[1] > ref_image.shape[0] else False

    scaled_ref_images = [
        proportional_resize(ref_image, res if scale_x else None, None if scale_x else res, occupation_fraction, interpolation=cv2.INTER_AREA) 
        for res in resolutions
    ]

    lod_images = []
    for res in resolutions:
        if(os.path.isfile(file_prefix + "_lod_{}{}".format(res,hdr_extension))):
            lod_images.append(image.lin_to_srgb(sio.read(file_prefix + "_lod_{}{}".format(res,hdr_extension))))
        else:
            print("Resolution {} not found, using blank image".format(res))
            lod_images.append(img_empty)

    scaled_lod_images = [
        proportional_resize(img, res if scale_x else None, None if scale_x else res, occupation_fraction, interpolation=cv2.INTER_AREA) 
        for (res,img) in zip(resolutions,lod_images)
    ]

    error_images = [
        compute_flip_error(lod,ref,background_color=[1.0,1.0,1.0])
        #compute_relMSE_error(lod,ref,dynamic_range=14, exposure=10)
        for (ref,lod) in zip(scaled_lod_images,scaled_ref_images)
    ]

    return ref_image, scaled_ref_images, lod_images, scaled_lod_images, error_images

parser = argparse.ArgumentParser(description='Lod Comparison Figure generator')
parser.add_argument("scene_base_path", help="scene base path without any suffix or extension")
parser.add_argument('--angle', type=int, default=10, help='angle for image split (default: 10)')
parser.add_argument('--max_res', type=int, default=512, help='maximum lod resolution (in voxels) (default: 512)')
parser.add_argument('--num_lods', type=int, default=6, help='number of processed lods (default: 6)')
parser.add_argument('--occupation_fraction', type=float, default=1.0, help='fraction of the image covered by the render in the smallest axis (default: 1.0)')
parser.add_argument('--skip_out', action='store_true', help='disable our lod output')
parser.add_argument('--postfix_string', type=str, default="", help='string added to output file (default: "")')

args = parser.parse_args()

file_prefix = args.scene_base_path
scene_name = os.path.basename(os.path.normpath(file_prefix))
scene_name += args.postfix_string
input_degree = args.angle

num_lods = args.num_lods
max_pow = int(np.log2(args.max_res))
min_pow = max_pow - num_lods + 1
resolutions = [2**i for i in range(min_pow,max_pow + 1)]

print("resolutions : ",resolutions)

ref_image, scaled_ref_images, lod_images, scaled_lod_images, error_images = load_images(file_prefix,resolutions,args.occupation_fraction)

# check if loubet's hybrid technique is availabee
ref_loubet_prefix = os.path.dirname(file_prefix) + os.path.sep + "loubet_" + os.path.basename(os.path.dirname(file_prefix))
add_loubet = os.path.isfile(ref_loubet_prefix + "_ref.exr")
if add_loubet :
    print("Adding comparison with Loubet's work")
    loubet_ref_image, loubet_scaled_ref_images, loubet_lod_images, loubet_scaled_lod_images, loubet_error_images = load_images(ref_loubet_prefix,resolutions,args.occupation_fraction, '.exr')

vertical_split = False if abs(input_degree) > 45 else True
if(not vertical_split):
    input_degree = 90 - input_degree if input_degree > 0 else -(input_degree + 90) 
images = [
    [image.SplitImage([ref_image, lod_images[i]], degree=input_degree, weights=[0.55,0.45], vertical=vertical_split) for i in range(0,num_lods)],
    [image.SplitImage([scaled_ref_images[i], scaled_lod_images[i]], degree=input_degree, weights=[0.55,0.45], vertical=vertical_split) for i in range(0,num_lods)],
    [error_images[i] for i in range(0,num_lods)]
]

if args.skip_out:
    print("Skipping lod output row")
    images.pop(0)

if add_loubet : 
    images.append([image.SplitImage([loubet_scaled_ref_images[i], loubet_scaled_lod_images[i]], degree=input_degree, weights=[0.55,0.45], vertical=vertical_split) for i in range(0,num_lods)])
    images.append([loubet_error_images[i] for i in range(0,num_lods)])

# ---------- Simple Grid with SplitImages ----------
n_rows = len(images)
top_cols = num_lods
top_grid = figuregen.Grid(num_rows=n_rows, num_cols=top_cols)

# fill grid with image data
for row in range(n_rows):
    for col in range(top_cols):
        s_img = images[row][col]
        try:
            raw_img = figuregen.PNG(s_img)
        except:
            raw_img = figuregen.PNG(s_img.get_image())
        e = top_grid.get_element(row,col).set_image(raw_img)
        try:
            e.draw_lines(s_img.get_start_positions(), s_img.get_end_positions(), linewidth_pt=0.5, color=[0,0,0])
        except:
            e.set_frame(0.5)

top_grid.set_col_titles('top', ["Ref/Lod {}".format(i) for i in reversed(range(0,num_lods))])

row_titles = ['Output','At Scale','Flip Error']

if args.skip_out:
    row_titles.pop(0)

if add_loubet :
    row_titles += ['At Scale Loubet',"Flip Error Loubet"]
top_grid.set_row_titles('left', row_titles)

top_grid.set_title('top', r"\textbf{" + scene_name.replace("_"," ") + "}")

# LAYOUT: Specify paddings (unit: mm)
top_lay = top_grid.get_layout()
top_lay.set_padding(column=1.0, right=1.5,bottom=1.0)
top_lay.set_col_titles('top', field_size_mm=5.0)

if __name__ == "__main__":
    width_cm = 15
    basename = file_prefix + "_comparison_rescaled" + args.postfix_string
    figuregen.figure([[top_grid]], width_cm=width_cm, filename=basename+'.pdf',
        backend=figuregen.PdfBackend(intermediate_dir="log"))
    try:
        from figuregen.util import jupyter
        jupyter.convert(basename+'.pdf', 600)
        os.remove(basename+'.pdf')
    except:
        print('Warning: pdf could not be converted to png')


# ---------- Simple Grid with Side by side comparison----------
images_side_by_side = [
    [lod_images[i] for i in range(0,num_lods)],
    [scaled_ref_images[i] for i in range(0,num_lods)],
    [scaled_lod_images[i] for i in range(0,num_lods)],
    [error_images[i] for i in range(0,num_lods)]
]

if args.skip_out:
    images_side_by_side.pop(0)

if add_loubet : 
    images_side_by_side.append([loubet_scaled_lod_images[i] for i in range(0,num_lods)])
    images_side_by_side.append([loubet_error_images[i] for i in range(0,num_lods)])

n_rows = len(images_side_by_side)
top_cols = num_lods
top_grid = figuregen.Grid(num_rows=n_rows, num_cols=top_cols)

# fill grid with image data
for row in range(n_rows):
    for col in range(top_cols):
        s_img = images_side_by_side[row][col]
        raw_img = figuregen.PNG(s_img)
        e = top_grid.get_element(row,col).set_image(raw_img)
        e.set_frame(0.5)

top_grid.set_col_titles('top', ["Lod {}".format(i) for i in reversed(range(0,num_lods))])
row_titles = ['Out','Ref','LoD','Flip Error']

if args.skip_out:
    row_titles.pop(0)
    
if add_loubet :
    row_titles += ['Lod Loubet',"Flip Error Loubet"]
top_grid.set_row_titles('left',row_titles)

top_grid.set_title('top', r"\textbf{" + scene_name.replace("_"," ") + "}")

# LAYOUT: Specify paddings (unit: mm)
top_lay = top_grid.get_layout()
top_lay.set_padding(column=1.0, right=1.5,bottom=1.0)
top_lay.set_col_titles('top', field_size_mm=5.0)

if __name__ == "__main__":
    width_cm = 20
    basename = file_prefix + "_comparison_rescaled_side_by_side" + args.postfix_string
    figuregen.figure([[top_grid]], width_cm=width_cm, filename=basename+'.pdf',
        backend=figuregen.PdfBackend(intermediate_dir="log"))
    try:
        from figuregen.util import jupyter
        jupyter.convert(basename+'.pdf', 600)
        os.remove(basename+'.pdf')
    except:
        print('Warning: pdf could not be converted to png')

