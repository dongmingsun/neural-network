import numpy as np
import copy
from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)


def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    W = weight_scale * np.random.randn(n_in, n_out)
    b = np.zeros(n_out)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b


class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """

    def __init__(self,
                 hidden_dims,
                 input_dim=32 * 32 * 3,
                 num_classes=10,
                 dropout=0,
                 reg=0.0,
                 weight_scale=1e-2,
                 dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: An integer giving the size of the input
        - num_classes: An integer giving Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.
        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        self.grads_ = dict()
        self.last_grads_ = dict()
        self.gij_ = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # dims is: [15 20 30 10]
        dims = np.hstack((input_dim, hidden_dims, num_classes))
        for i in range(self.num_layers):
            # weight size: n_in * n_out, bias size: n_out
            self.params['W%d' % (i + 1)], self.params['b%d' % (i + 1)] = random_init(dims[i], dims[i + 1], weight_scale,
                                                                                     dtype)
            self.gij_['W%d' % (i + 1)], self.gij_['b%d' % (i + 1)] = np.ones((dims[i], dims[i + 1]),
                                                                               dtype=np.float), np.ones(dims[i + 1])
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
            else:
                self.dropout_params["seed"] = None

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        x_in = X
        # If y is None then we are in test mode so just return scores
        if y is None:
            for i in range(1, self.num_layers):
                linear_out = linear_forward(x_in, self.params['W' + str(i)], self.params['b' + str(i)])
                relu_out = relu_forward(linear_out)
                x_in = relu_out
            logits = linear_forward(x_in, self.params['W' + str(self.num_layers)],
                                    self.params['b' + str(self.num_layers)])
            scores = softmax(logits, y)
            # If y is not None then we are in train mode so just return scores
            return scores
        else:
            self.last_grads_ = copy.deepcopy(self.grads_)
            loss, grads = 0, dict()
            self.grads_ = grads
            for i in range(1, self.num_layers):
                linear_out = linear_forward(x_in, self.params['W' + str(i)], self.params['b' + str(i)])
                linear_cache["out" + str(i)] = linear_out
                relu_out = relu_forward(linear_out)
                relu_cache["out" + str(i)] = relu_out
                dropout_out = relu_out
                if self.use_dropout:
                    seed = 42 if self.dropout_params["seed"] is None else self.dropout_params["seed"]
                    dropout_out, mask = dropout_forward(relu_out, self.dropout_params["p"], True, seed)
                    dropout_cache["out" + str(i)] = dropout_out
                    dropout_cache["mask" + str(i)] = mask
                x_in = dropout_out
            logits = linear_forward(x_in, self.params['W' + str(self.num_layers)],
                                    self.params['b' + str(self.num_layers)])
            loss, dlogits = softmax(logits, y)

            #######################################################################
            #                            END OF YOUR CODE                         #
            #######################################################################
            # """
            # TODO: Implement the backward pass for the fully-connected net. Store
            # the loss in the loss variable and all gradients in the grads
            # dictionary. Compute the loss with softmax. grads[k] has the gradients
            # for self.params[k]. Add L2 regularisation to the loss function.
            # NOTE: To ensure that your implementation matches ours and you pass the
            # automated tests, make sure that your L2 regularization includes a
            # factor of 0.5 to simplify the expression for the gradient.
            # """
            #######################################################################
            #                           BEGIN OF YOUR CODE                        #
            #######################################################################

            dout = dlogits
            if self.use_dropout:
                input_x = dropout_cache["out" + str(self.num_layers - 1)]
            else:
                input_x = relu_cache["out" + str(self.num_layers - 1)]
            W = self.params['W' + str(self.num_layers)]
            b = self.params['b' + str(self.num_layers)]
            dout, dW, db = linear_backward(dout, input_x, W, b)
            grads['W' + str(self.num_layers)] = dW
            grads['b' + str(self.num_layers)] = db
            for i in range(self.num_layers - 1, 1, -1):
                if self.use_dropout:
                    mask = dropout_cache['mask' + str(i)]
                    dout = dropout_backward(dout, mask, seed, self.dropout_params["p"])
                input_x = linear_cache["out" + str(i)]
                dout = relu_backward(dout, input_x)
                if self.use_dropout:
                    input_x = dropout_cache["out" + str(i - 1)]
                else:
                    input_x = relu_cache["out" + str(i - 1)]
                W = self.params['W' + str(i)]
                b = self.params['b' + str(i)]
                dout, dW, db = linear_backward(dout, input_x, W, b)
                grads['W' + str(i)] = dW
                grads['b' + str(i)] = db
            if self.use_dropout:
                mask = dropout_cache['mask' + str(1)]
                dout = dropout_backward(dout, mask, seed,
                                        self.dropout_params["p"])
            input_x = linear_cache["out" + str(1)]
            dout = relu_backward(dout, input_x)
            W = self.params['W' + str(1)]
            b = self.params['b' + str(1)]
            dout, dW, db = linear_backward(dout, X, W, b)
            grads['W' + str(1)] = dW
            grads['b' + str(1)] = db

            # add L2 regularisation
            regularisation = 0.0
            for i in range(1, self.num_layers + 1):
                tempW = 0.5 * self.reg * np.square(self.params['W' + str(i)])
                regularisation += np.sum(tempW)
                grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
            loss += regularisation
            #######################################################################
            #                            END OF YOUR CODE                         #
            #######################################################################
            return loss, grads
