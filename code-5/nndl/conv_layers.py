import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape # N Cifar10 images of shape H,W,C = 32,32, 3
  F, _, HH, WW = w.shape # number of filters, and receptive field 
  stride = conv_param.get('stride', 1)
  pad = conv_param.get('pad', 0)
    
  H_prime = 1 + (H - HH + 2*pad) // stride #output size, depth will be F
  W_prime = 1 + (W - WW + 2*pad) // stride
    
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0) #pads by pad rows of 0s along all four dims. 
  out = np.zeros((N, F, H_prime, W_prime)) #set up output array
  

  #naive implmenentation: using four for loops to set the value of every single neuron in the ouput array, 
  for n in np.arange(N): #iterate over number of images
    for f in np.arange(F): #iterate over number of filters
        for j in np.arange(0, H_prime): #iterate over height of output array
            for i in np.arange(0, W_prime): #iterate over width of output array
                out[n, f, j, i] = (x_pad[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW] * w[f, :, :, :]).sum() + b[f]
                #convolution implmented as localied elementwise multiplication between filter and zero padded input matrix
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  # Extract shapes and constants
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride = conv_param.get('stride', 1)
  pad = conv_param.get('pad', 0)
  # Padding
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
  H_prime = 1 + (H + 2 * pad - HH) // stride
  W_prime = 1 + (W + 2 * pad - WW) // stride
  # Construct output
  dx_pad = np.zeros_like(x_pad)
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  # Naive Loops
  for n in range(N):
    for f in range(F):
      db[f] += dout[n, f].sum()
      for j in range(0, H_prime):
        for i in range(0, W_prime):
          dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dout[n, f, j, i]
          dx_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f] * dout[n, f, j, i]
    # Extract dx from dx_pad
  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  HH = pool_param.get('pool_height', 2)
  WW = pool_param.get('pool_width', 2)
  stride = pool_param.get('stride', 2)
  H_prime = 1 + (H - HH) // stride
  W_prime = 1 + (W - WW) // stride
  out = np.zeros((N, C, H_prime, W_prime))
  for n in range(N):
    for j in range(H_prime):
      for i in range(W_prime):
        out[n, :, j, i] = np.amax(x[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW], axis=(-1, -2))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  x, pool_param = cache
  N, C, H, W = x.shape
  HH = pool_param.get('pool_height', 2)
  WW = pool_param.get('pool_width', 2)
  stride = pool_param.get('stride', 2)
  H_prime = 1 + (H - HH) // stride
  W_prime = 1 + (W - WW) // stride
  # Construct output
  dx = np.zeros_like(x)
  # Naive Loops
  for n in range(N):
    for c in range(C):
      for j in range(H_prime):
        for i in range(W_prime):
          ind = np.argmax(x[n, c, j*stride:j*stride+HH, i*stride:i*stride+WW])
          ind1, ind2 = np.unravel_index(ind, (HH, WW))
          dx[n, c, j*stride:j*stride+HH, i*stride:i*stride+WW][ind1, ind2] = dout[n, c, j, i]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape
  running_mean = bn_param.get('running_mean', np.zeros((1, C, 1, 1), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((1, C, 1, 1), dtype=x.dtype))

  if mode == 'train':
    mu = np.mean(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    var = 1 / float(N * H * W) * np.sum((x - mu) ** 2, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
    out = y

    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var

    cache = (x_hat, mu, var, eps, gamma, beta, x)

  elif mode == 'test':
    x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    y = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
    out = y

  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  x_hat, mu, var, eps, gamma, beta, x = cache
  N, C, H, W = dout.shape

  dbeta = np.sum(dout, axis=(0, 2, 3))
  dgamma = np.sum(dout * x_hat, axis=(0, 2, 3))

  gamma_reshape = gamma.reshape(1, C, 1, 1)
  beta_reshape = beta.reshape(1, C, 1, 1)
  Nt = N * H * W

  dx_hat = dout * gamma_reshape
  dxmu1 = dx_hat * 1 / np.sqrt(var + eps)
  divar = np.sum(dx_hat * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1)
  dvar = divar * -1 / 2 * (var + eps) ** (-3 / 2)
  dsq = 1 / Nt * np.broadcast_to(np.broadcast_to(np.squeeze(dvar), (W, H, C)).transpose(2, 1, 0), (N, C, H, W))
  dxmu2 = 2 * (x - mu) * dsq
  dx1 = dxmu1 + dxmu2
  dmu = -1 * np.sum(dxmu1 + dxmu2, axis=(0, 2, 3))
  dx2 = 1 / Nt * np.broadcast_to(np.broadcast_to(np.squeeze(dmu), (W, H, C)).transpose(2, 1, 0), (N, C, H, W))
  dx = dx1 + dx2

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta