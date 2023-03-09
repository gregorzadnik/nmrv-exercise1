import numpy as np
from ex1_utils import *
from scipy import signal

def lucaskanade(im1, im2, N):

    im1 = gausssmooth(im1, 1)/255
    im2 = gausssmooth(im2, 1)/255


    # First calculate the temporal, x, and y derivatives
    Dt = gausssmooth(im2 - im1, 1)    
    Dx, Dy = gaussderiv((im1+im2)/2, 1)

    kernel = np.ones((N, N))


    Dx2 = signal.convolve2d(np.multiply(Dx, Dx), kernel, mode='same')
    Dy2 = signal.convolve2d(np.multiply(Dy, Dy), kernel, mode='same')
    Dxt = signal.convolve2d(np.multiply(Dx, Dt), kernel, mode='same')
    Dyt = signal.convolve2d(np.multiply(Dy, Dt), kernel, mode='same')
    Dxy = signal.convolve2d(np.multiply(Dx, Dy), kernel, mode='same')


    denominator = Dx2*Dy2 - Dxy*Dxy
    denominator += 1e-5
    
    u = -(Dy2*Dxt - Dxy*Dyt)/denominator
    v = -(Dx2*Dyt - Dxy*Dxt)/denominator
    return u, v