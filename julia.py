#!/env/bin/python

import os, sys, errno
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel


complex_gpu = ElementwiseKernel(
    "pycuda::complex<float> *q, int *output, float creal, float cimag",
    """
    {
        float zimag = q[i].imag();
        float zreal = q[i].real();
        float zreal2 = zreal * zreal;
        float zimag2 = zimag * zimag;
        int iter = 0;

        output[i] = 0;
        while (zimag2 + zreal2 <= 4) {
            if (iter++ > 750) break;
            zimag = 2 * zreal * zimag + cimag;
            zreal = zreal2 - zimag2 + creal;
            zreal2 = zreal * zreal;
            zimag2 = zimag * zimag;
            output[i] += 1;
        }
    }
    """,
    "complex5",
    preamble="#include <pycuda-complex.hpp>",)


def init_gpu(xf, xt, yf, yt, w, h):
    r1 = np.linspace(xf, xt, w, dtype=np.float32)
    r2 = np.linspace(yf, yt, h, dtype=np.float32)
    c = r1 + r2[:,None]*1j

    q_gpu = gpuarray.to_gpu(c.astype(np.complex64))
    iterations_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))

    return q_gpu, iterations_gpu


def julia_set(q_gpu, iterations_gpu, c_real, c_imag):
    complex_gpu(q_gpu, iterations_gpu, c_real, c_imag)
    return iterations_gpu.get()


def to_image(outf, z, w, h, dpi, cmap, norm, plotc):
    norm = colors.PowerNorm(norm)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi, frameon=False)
    fig.patch.set_visible(False)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') 

    if plotc:
        ax.text(0.95, 0.01, "{}, {}".format(plotc[0], plotc[1]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='white', fontsize=40)

    ax.imshow(z, cmap=cmap, norm=norm)

    plt.savefig(outf, format="png", dpi=dpi)
    plt.close('all')


def out(message):
    sys.stdout.write("Frame: %d of %d \r" % (n, args.frames) )
    sys.stdout.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dpi',    type=int,   help='Image density',    required=False, default=30)
    parser.add_argument('--width',  type=int,   help='Image width',      required=False, default=16)
    parser.add_argument('--height', type=int,   help='Image height',     required=False, default=9)
    parser.add_argument('--frames', type=int,   help='Number of frames', required=False, default=100)
    parser.add_argument('--stepr',  type=float, help='Real Step',        required=False, default=None)
    parser.add_argument('--stepi',  type=float, help='Imaginary Step',   required=False, default=None)
    parser.add_argument('--real',   type=float, help='Starting point',   required=False, default=-0.74)
    parser.add_argument('--imag',   type=float, help='Starting point',   required=False, default=0.16)
    parser.add_argument('--norm',   type=float, help='Color normal',     required=False, default=0.4)
    parser.add_argument('--dir',    type=str,   help='Output directory', required=True,  default="out")
    parser.add_argument('--prefix', type=str,   help='Output prefix',    required=False, default="julia")
    parser.add_argument('--cmap',   type=str,   help='Color map',        required=False, default="gnuplot2")
    parser.add_argument('--xfrom',  type=float, help='X start',          required=False, default=-2.0)
    parser.add_argument('--xto',    type=float, help='X end',            required=False, default=2.0)
    parser.add_argument('--yfrom',  type=float, help='Y start',          required=False, default=-1.0)
    parser.add_argument('--yto',    type=float, help='Y end',            required=False, default=1.0)
    parser.add_argument('--plotc',  action='store_true', help='Plot c values', default=False)

    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    q_gpu, iterations_gpu = init_gpu(
        args.xfrom, args.xto, args.yfrom, args.yto, 
        args.dpi * args.width, args.dpi * args.height)

    c_real = args.real
    c_imag = args.imag
    out_tpl = os.path.join(args.dir, "".join([args.prefix, "_{}.png"]))

    for n in range(1, args.frames + 1):
        out("Frame: {} of {} \r".format(n, args.frames))
        z = julia_set(q_gpu, iterations_gpu, c_real, c_imag)
        outf = out_tpl.format(str(n).zfill(len(str(args.frames))))

        to_image(
            outf, z, args.width, args.height, args.dpi, args.cmap,
            args.norm, [c_real, c_imag] if args.plotc else False)

        if args.stepr is not None:
            c_real += args.stepr

        if args.stepi is not None:
            c_imag += args.stepi
