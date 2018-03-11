import numpy as np


def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: A numpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data


    """
    out = None
    """
    TODO: Implement the linear forward pass. Store your result in `out`.
    Tip: Think how X needs to be reshaped.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    """
    For each of the example X[i] in X, the shape
    (i.e. the tuple of dimension of each of the example) is (d_1,...d_k).
    We will reshape X[i] into a vector of dimension K = d_1*d_2_d3...*d_k
    Then calculate the output (of shape (N,M)) by dot multiplying X and W
    and adding b"""

    input_shape = X.shape[0]  # This is the original shape of an example
    new_X = X.reshape(input_shape, -1)  # We now resize X
    out = new_X.dot(W) + b

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def linear_backward(dout, X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d1, ..., d_k), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A nump array of shape (M,), gradient with respect to b
    """
    dX, dW, db = None, None, None
    """
    TODO: Implement the linear backward pass. Store your results of the
    gradients in `dX`, `dW`, `db`.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    dX = dout.dot(W.T).reshape(X.shape)
    dW = X.reshape(X.shape[0], -1).T.dot(dout)
    db = np.sum(dout, 0)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """
    out = None
    """
    TODO: Implement the ReLU forward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    out = X.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout
    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the ReLU backward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    dX = dout.copy()
    dX[np.where(X < 0)] = 0

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX


def dropout_forward(X, p=0.5, train=True, seed=42):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    out = None
    mask = None
    if seed:
        np.random.seed(seed)
    """
    TODO: Implement the inverted dropout forward pass. Make sure to consider
    both train and test case. Pay attention scaling the activation function.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    if train is False:
        out = X
    else:
        """
        The mask will be a vector of Bernoulli distributed variables.
        With inverted dropout, we scale the input X by the probability of keeping.
        Essentially this scaled input the original input divided by (1-p)
        so that we do not have to multiply (1-p) when we test.
        """
        # generate random number from 0 to 1 and if the number is larger to
        # the probablity of dropping, set corresponding mask element to 1, otherwise, mask is 0
        mask = (np.random.rand(*X.shape) > p) / 1
        out = X * mask / (1 - p)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out, mask


def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the inverted backward pass for dropout. Make sure to
    consider both train and test case.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    if train is True:
        dX = dout * mask / (1 - p)
    else:
        dX = dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX
