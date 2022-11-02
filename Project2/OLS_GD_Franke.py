import autograd.numpy as jnp
import numpy as nppip
from jax import grad, jit, vmap

""""
In summary, you should perform an analysis of the results for OLS and Ridge
regression as function of the chosen learning rates, the number of mini-batches
and epochs as well as algorithm for scaling the learning rate. 
i.e:

    MSE vs learning rate 
    MSE vs mini batches 
    MSE vs epochs 
    MSE vs complexity for GD, GD w m, SGD, SGD w m 


    GD_Franke: Code for GD on Franke function w/o momentum and fixed learning rate
    GDM_Franke: Code for GD on Franke function w momentum and fixed learning rate - compare to above
    SGD_Franke: Code for SGD on Franke function w/o momentum, fixed learning rate, w epochs, mini batches 
    SGDM_Franke: Code for SGD on Franke function w momentum, fixed learning rate, w epochs, mini batches
    AG_GD_Franke:  --"-- with autograd
    AG_GDM_Franke:  --"-- with autograd
    AG_SDG_Franke:  --"-- with autograd
    AG_SDGM_Franke:  --"-- with autograd
    add RMSProp and ADAM
"""

#random seed 
np.random.seed(2021)
#meshgrid of datapoints
#test & train split 
#noise?
#polynomial degree

