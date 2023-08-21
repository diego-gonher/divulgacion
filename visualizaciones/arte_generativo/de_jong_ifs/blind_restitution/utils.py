import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from numba import jit
from math import sin, cos, sqrt, fabs
from datashader.utils import export_image
from PIL import Image, ImageOps, ImageEnhance
from typing import Callable
from datetime import date


# ATTRACTORS
@jit(nopython=True)
def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), \
           sin(b * x) + d * cos(b * y)


@jit(nopython=True)
def De_Jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), \
           sin(c * x) - cos(d * y)


@jit(nopython=True)
def De_Jong3(x, y, a, b, c, d, *o):
    return sin(a * y)**3 - cos(b * x), \
           sin(c * x) - 5*cos(d * y)


@jit(nopython=True)
def De_Jong5(x, y, a, b, c, d, *o):
    return sin(a * y**2) * cos(b * x)**2, \
           sin(c * x) - 5*cos(d * y**2)


@jit(nopython=True)
def De_Jong6(x, y, a, b, c, d, *o):
    return sin(a * y**3) * cos(b * x)**2, \
           sin(c * x) - 2*cos(d * y**2)


@jit(nopython=True)
def De_Jong7(x, y, a, b, c, d, *o):
    return sin(a * y**3) - cos(b * x)**3, \
           sin(c * x) * 2*cos(d * y**2)


@jit(nopython=True)
def De_Jong8(x, y, a, b, c, d, *o):
    return sin(a * y*x) - cos(b * x), \
           sin(c * x) - cos(d * y)


@jit(nopython=True)
def De_Jong9(x, y, a, b, c, d, *o):
    return sin(a * y**3) * cos(b * x)**3, \
           sin(c * x * y) - 3*cos(d * y**2)


@jit(nopython=True)
def De_Jong10(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), \
           sin(c * x * y**2) - 3*cos(d * y)


@jit(nopython=True)
def De_Jong11(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), \
           sin(c * x * y**5) - 4*cos(d * y)


@jit(nopython=True)
def Hopalong1(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), \
           a - x


@jit(nopython=True)
def G(x, mu):
    return mu * x + 2 * (1 - mu) * x**2 / (1.0 + x**2)


@jit(nopython=True)
def Gumowski_Mira(x, y, a, b, mu, *o):
    xn = y + a*(1 - b*y**2)*y  +  G(x, mu)
    yn = -x + G(xn, mu)
    return xn, yn


@jit(nopython=True)
def Fractal_Dream(x, y, a, b, c, d, *o):
    return sin(y*b)+c*sin(x*b), \
           sin(x*a)+d*sin(y*a)


# DATA CREATION
@jit(nopython=True)
def trajectory_coords(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=50000000):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x,y


def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=50000000):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x, y=y))


# IMAGE MANIPULATION
# BORDER
def add_border(input_image, output_image, border, color=0):
    img = Image.open(input_image)
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
    bimg.save(output_image)


# PIXEL SORTING
def sort_pixels(image: Image, value: Callable, condition: Callable, rotation: int = np.pi) -> Image:
    pixels = np.rot90(np.array(image), rotation)
    values = value(pixels)
    edges = np.apply_along_axis(lambda row: np.convolve(row, [-1, 1], 'same'), 0, condition(values))
    intervals = [np.flatnonzero(row) for row in edges]

    for row, key in enumerate(values):
        order = np.split(key, intervals[row])
        for index, interval in enumerate(order[1:]):
            order[index + 1] = np.argsort(interval) + intervals[row][index]
        order[0] = range(order[0].size)
        order = np.concatenate(order)

        for channel in range(3):
            pixels[row, :, channel] = pixels[row, order.astype('uint32'), channel]

    return Image.fromarray(np.rot90(pixels, -rotation))


def luminosity(pixels):
    return np.average(pixels, axis=2) / 255


def hue(pixels):
    r, g, b = np.split(pixels, 3, 2)
    return np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)[:, :, 0]


def sat(pixels):
    r, g, b = np.split(pixels, 3, 2)
    maximum = np.maximum(r, np.maximum(g, b))
    minimum = np.minimum(r, np.minimum(g, b))
    return ((maximum - minimum) / maximum)[:, :, 0]


def laplace(pixels):
    from scipy.signal import convolve2d
    lum = np.average(pixels, 2) / 255
    return np.abs(convolve2d(lum, np.array([[0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]]), 'same'))


class MyCMAPS:
    def __init__(self):
        """
        This class contains my custom cmaps for datashader
        """
        self.cmap_pink = ['#FFFFFF', '#FF8FB1', '#B270A2', '#7A4495']
        self.cmap_eternal_blue = ['#FFFFFF', '#8e9fa7', '#848c90', '#106076', '#188ca4', '#7cd4dc']
        self.cmap_red_highlight = ['#FFFFFF', '#000000', '#820000']
        self.cmap_red_highlight2 = ['#FFFFFF', '#000000', '#FF0000']
        self.cmap_plane = ['#DDDDDD', '#414141', '#30475E', '#FF0000']
        self.cmap_fall = ['#FFFFFF', '#603601', '#361500', '#1C0A00']
        self.cmap_bw = ['#FFFFFF', 'black']
        self.cmap_bw_r = ['black', '#FFFFFF']
        self.cmap_gold_ren = ['#FFFFFF', '#D7A86E', '#A64B2A', '#8E3200', '#E6B325', '#BF9742', '#A47E3B']
        self.cmap_starry_night = ['#FFFFFF', '#0B1E38', '#DB901C', '#E8E163', '#7FC5DC', '#4888C8', '#173679']
        self.cmap_elegant_love = ['#FFFFFF', '#680819', '#800021', '#9F1D34', '#D1B12F', '#CC9900']
        self.cmap_mona_lisa = ['#FFFFFF', '#727F4B', '#A9A569', '#E9C468', '#92692E', '#764B1C', '#352524']
        self.cmap_liberty_leading = ['#FFFFFF', '#fffed8', '#463b32', '#7a6c5a', '#97876d', '#3f4b58',
                                     '#1e3049', '#9e2721']
        self.cmap_dante_virgil = ['#FFFFFF', '#8c6d46', '#40240c', '#d9ab82', '#8c4b26', '#400904']
        self.cmap_dulle_griet = ['#FFFFFF', '#af3a4a', '#242425', '#2c2929', '#6c2f24', '#AF5853']
        self.cmap_the_way_it_ends = ['#FFFFFF', '#252c44', '#3d343d', '#8c9ca9', '#4c494f', '#DEC066']
        self.cmap_alien = ['#FFFFFF', '#dcdee0', '#3c6c8c', '#74245c', '#794a72', '#edb2b7']
        self.cmap_will_of = ['#FFFFFF', '#f0862c', '#e96411', '#961b0b', '#657A79', '#AAA1A2']
        self.cmap_typhoons = ['#FFFFFF', '#2F292D', '#112b31', '#115566', '#00B9BF', '#EE3956', '#f47d8e']
        self.cmap_death_we = ['#FFFFFF', '#968B79', '#E6E9D6', '#525150', '#343c3b', '#424c44', '#68401c', '#C49867',
                              '#D9BA83']
        self.cmap_vibrations = ['#FFFFFF', '#706260', '#6c636c', '#BBAAB0', '#684932', '#4f3f3e', '#684932', '#954461',
                                '#B85F7D', '#D7AF79']