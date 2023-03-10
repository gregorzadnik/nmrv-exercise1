import numpy as np
from ex1_utils import *
from scipy import signal
from lucas_kanade import lucaskanade

def hornschunck(im1, im2, n_iters, lmbd, use_lk=False):
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    if(use_lk):
        u, v = lucaskanade(im1, im2, 3)

    im1 = gausssmooth(im1, 1)/255
    im2 = gausssmooth(im2, 1)/255

    Dt = gausssmooth(im2 - im1, 1)    
    Dx, Dy = gaussderiv((im1+im2)/2, 1)

    ld = np.array([[0, 0.25, 0], 
                   [0.25, 0, 0.25], 
                   [0, 0.25, 0]])
    
    D = lmbd + Dx*Dx + Dy*Dy
    for i in range(n_iters):
        ua = signal.convolve2d(u, ld, mode='same')
        va = signal.convolve2d(v, ld, mode='same')

        P = Dx*ua + Dy*va + Dt
        fract = P/D
        u = ua - Dx*fract
        v = va - Dy*fract

    return u, v