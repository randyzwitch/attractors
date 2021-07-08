# this example adapted from https://examples.pyviz.org/attractors/attractors.html

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from numba import jit
from math import sin, cos, sqrt, fabs

from colorcet import palette

ds.transfer_functions.Image.border=0

##############################################
# "Front page" and Global content
st.set_page_config(page_title="Visualizing Attractors")


"# Visualizing Attractors"

"""An [attractor](https://en.wikipedia.org/wiki/Attractor#Strange_attractor) is a set of values to which a numerical system tends to evolve. 
An attractor is called a [strange attractor](https://en.wikipedia.org/wiki/Attractor#Strange_attractor) if the resulting pattern has a fractal structure.

This Streamlit app (adapted from [this example notebook](https://examples.pyviz.org/attractors/attractors.html) from [PyViz](https://examples.pyviz.org/index.html#)) shows how to calculate and plot two-dimensional attractors of a variety of types, using code and parameters primarily 
from [L&aacute;zaro Alonso](https://lazarusa.github.io/Webpage/codepython2.html), 
[François Pacull](https://aetperf.github.io/2018/08/29/Plotting-Hopalong-attractor-with-Datashader-and-Numba.html), 
[Jason Rampe](https://softologyblog.wordpress.com/2017/03/04/2d-strange-attractors), [Paul Bourke](http://paulbourke.net/fractals/), 
and [James A. Bednar](http://github.io/jbednar)."""

"""
For example, a [Clifford Attractor](http://paulbourke.net/fractals/clifford) is a strange attractor 
defined by two iterative equations that determine the _x,y_ locations of discrete steps in the path of a 
particle across a 2D space, given a starting point _(x0,y0)_ and the values of four parameters _(a,b,c,d)_:
"""

st.latex("x_{n +1} = \sin(a y_{n}) + c \cos(a x_{n})")

st.latex("y_{n +1} = \sin(b x_{n}) + d \cos(b y_{n})")

"""
At each time step, the equations define the location for the following time step, and the accumulated locations show the areas of the 2D plane most commonly visited by the imaginary particle.  
"""

"""---"""

##############################################

# sidebar options
st.sidebar.markdown("Global options")
n = st.sidebar.number_input("# of obs", 10 ** 6, 10 ** 7, step=10 ** 6)
plotsize = st.sidebar.number_input("plot size", 500, 700, step=50)
cmap = st.sidebar.selectbox(
    "color map",
    ("bgy", "bmw", "bgyw", "bmy", "fire", "gray", "kgy", "kbc"),
)
st.sidebar.markdown("---")
st.sidebar.markdown("Attractor and parameters")
st.sidebar.markdown(
    """_Toggle the choices to see how the attractor changes with different inputs_"""
)
fn = st.sidebar.selectbox("Attractor", ["Clifford", "De Jong", "Svensson"])
a = st.sidebar.number_input("a", -2.0, 2.0, -1.3, step=0.1)
b = st.sidebar.number_input("b", -2.0, 2.0, -1.3, step=0.1)
c = st.sidebar.number_input("c", -2.0, 2.0, -1.8, step=0.1)
d = st.sidebar.number_input("d", -2.0, 2.0, -1.9, step=0.1)


@jit(nopython=True)
def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)


@jit(nopython=True)
def De_Jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)


@jit(nopython=True)
def Svensson(x, y, a, b, c, d, *o):
    return d * sin(a * x) - sin(b * y), c * cos(a * x) + cos(b * y)


@jit(nopython=True)
def trajectory_coords(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x, y


@st.cache(show_spinner=False)
def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x, y=y))


def dsplot(fn, vals, plotsize=plotsize, n=n, cmap=viridis, label=True):
    """Return a Datashader image by collecting `n` trajectory points for the given attractor `fn`"""
    lab = ("{}, " * (len(vals) - 1) + " {}").format(*vals) if label else None
    df = trajectory(fn, *vals, n=n)
    cvs = ds.Canvas(plot_width=plotsize, plot_height=plotsize)
    agg = cvs.points(df, "x", "y")
    img = tf.shade(agg, cmap=cmap, name=lab)
    return img


if fn == "Clifford":

    st.markdown("""## Clifford Attractor""")

    st.latex("x_{n +1} = \sin(%2.1f y_{n}) + %2.1f \cos(%2.1f x_{n})" % (a, c, a))
    st.latex("y_{n +1} = \sin(%2.1f x_{n}) + %2.1f \cos(%2.1f y_{n})" % (b, d, b))

    g = dsplot(Clifford, (0, 0, a, b, c, d), plotsize, cmap=palette[cmap][::-1])
    components.html(g._repr_html_(), height=plotsize + 10, width=plotsize + 10)

elif fn == "De Jong":

    st.markdown("""## De Jong Attractor""")

    st.latex("x_{n +1} = \sin(%2.1f y_{n}) - \cos(%2.1f x_{n})" % (a, b))
    st.latex("y_{n +1} = \sin(%2.1f x_{n}) - \cos(%2.1f y_{n})" % (c, d))

    dg = dsplot(De_Jong, (0, 0, a, b, c, d), plotsize, cmap=palette[cmap][::-1])
    components.html(dg._repr_html_(), height=plotsize + 10, width=plotsize + 10)

elif fn == "Svensson":

    st.markdown("""## Svensson Attractor""")

    st.latex("x_{n +1} = %2.1f * \sin(%2.1f x_{n}) - \sin(%2.1f y_{n})" % (d, a, b))
    st.latex("y_{n +1} = %2.1f * \cos(%2.1f x_{n}) + \cos(%2.1f y_{n})" % (c, a, b))

    sv = dsplot(Svensson, (0, 0, a, b, c, d), plotsize, cmap=palette[cmap][::-1])
    components.html(sv._repr_html_(), height=plotsize + 10, width=plotsize + 10)
