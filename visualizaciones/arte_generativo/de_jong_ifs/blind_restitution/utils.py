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