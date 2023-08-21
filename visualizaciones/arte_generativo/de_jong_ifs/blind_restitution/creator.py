import numpy as np
import pandas as pd
import datashader as ds
import matplotlib.pyplot as plt
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from numba import jit
from math import sin, cos, sqrt, fabs
from datashader.utils import export_image
from PIL import Image, ImageOps, ImageEnhance
from typing import Callable
from datetime import date
from IPython import embed

from utils import *

# creation of attractor
# LOGO De_Jong -1.1869514241230223, 0.8886138736437754, -2.8472353600232863, -1.9643008196986904  #
attr_fun = Hopalong1
a, b, c, d = np.random.uniform(-3, 3, 4)

N = 100000000
df = trajectory(attr_fun, 0, 0, a, b, c, n=N)  # trajectory(attr_fun, 0, 0, a, b, c, d, n=N)
cvs = ds.Canvas(plot_width=500, plot_height=500)
agg = cvs.points(df, 'x', 'y')
ds.transfer_functions.Image.border = 1

# image creation
cmaps = MyCMAPS()
my_cmap = cmaps.cmap_bw

# export image
img = tf.shade(agg, cmap=my_cmap)

plt.imshow(np.log(img), cmap='gray', origin='lower')
plt.show()

# embed()

# name
attractor_name = attr_fun.__name__
extension = 'png'
img_name = f'{date.today().strftime("%Y_%m_%d")}_{attractor_name}_a{a:.3f}_b{b:.3f}_c{c:.3f}_d{d:.3f}'

export_image(img, img_name, background="white", export_path="./output")

# add a border to the image
in_img = './output/' + img_name + f'.{extension}'
add_border(in_img,
           output_image=in_img,
           border=125,
           color='white')

# pixel sorting with luminosity
sort_pixels(Image.open(in_img).convert('RGB'),
            luminosity,
            lambda lum: (lum > 2 / 6) & (lum < 4 / 6), 1).save('./output/' + img_name + f'_PSlum.{extension}')

# embed()

# invert the original image
invert_image(in_img, './output/' + img_name + f'_inv.{extension}')

# invert the pixel sorted image
invert_image('./output/' + img_name + f'_PSlum.{extension}', './output/' + img_name + f'_PSlum_inv.{extension}')
