import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims) 
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    # ================================================================ #
    
    layer_in_dim = input_dim
    
    for i, h_dim in enumerate(hidden_dims): #num_layers = 3
      self.params['b%d' % (i+1)] = weight_scale * np.zeros(h_dim)
      self.params['W%d' % (i+1)] = weight_scale * np.random.randn(layer_in_dim, h_dim)
        
      if self.use_batchnorm:
        self.params['gamma%d' % (i+1)] = np.ones(h_dim)
        self.params['beta%d' % (i+1)] = np.zeros(h_dim)
      
      layer_in_dim = h_dim
      
    self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_in_dim, num_classes)
    self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)
    
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    # ================================================================ #

    layer = {} #dictionary to keep track of layer output
    layer[0] = X #the first layer is simply the input
    cache_layer = {} #dictionary to keep track of cache outputs from respective layers
    dp_cache = {} #dictionary to keep track of dropout cache from respective layers

    for i in np.arange(1, self.num_layers): #from layer 1 (not 0) till the last layer (not including the last layer)
      if self.use_batchnorm:
        layer[i], cache_layer[i] = affine_batch_relu_forward(layer[i-1], self.params['W%d' % i], self.params['b%d' % i],
                                                             self.params['gamma%d'%i], self.params['beta%d'%i],
                                                             self.bn_params[i-1])
      else:
        layer[i], cache_layer[i] = affine_relu_forward(layer[i - 1], self.params['W%d' % i], self.params['b%d' % i])
        
      if self.use_dropout:
        layer[i], dp_cache[i] = dropout_forward(layer[i], self.dropout_param)
    
    WLast = 'W%d' % self.num_layers
    bLast = 'b%d' % self.num_layers
    scores, cache_scores = affine_forward(layer[self.num_layers - 1], self.params[WLast], self.params[bLast])
    
    
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y) #loss gives loss on scores w.r.t y, dscores gives gradient

    for i in np.arange(1, self.num_layers + 1): ## add regularization loss to total loss
      loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)

    dx = {} #dictionary to hold backpropping gradients 
    dx[self.num_layers], grads[WLast], grads[bLast] = affine_backward(dscores, cache_scores) #holds the first upstream derivative
    grads[WLast] += self.reg * self.params[WLast] #regularization accounted for. 

    for i in reversed(np.arange(1, self.num_layers)): #for all previous layers
      if self.use_dropout:
        dx[i+1] = dropout_backward(dx[i+1], dp_cache[i])
      if self.use_batchnorm:
        dx[i], grads['W%d'%i], grads['b%d'%i], grads['gamma%d'%i], grads['beta%d'%i] = affine_batch_relu_backward(dx[i+1],
                                                                                                               cache_layer[i])
      else:
        dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1], cache_layer[i])
        
      grads['W%d' % i] += self.reg * self.params['W%d' % i]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return loss, grads
