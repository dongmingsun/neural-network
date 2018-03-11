import numpy as np
import matplotlib.pyplot as plt
import pickle  # HACK

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
t_data = get_CIFAR10_data()

# print(t_data["X_train"])
model = FullyConnectedNet(hidden_dims= [250,250],reg=0.5, weight_scale=1e-2)
solver = Solver(model, data=t_data,
                update_rule='sgd',
                optim_config= {
                  'learning_rate': 1e-3
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=50,
                print_every=100)
solver.train()

# plot
# plot
plt.subplot(2, 1, 1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel('Iteration')
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

# pickle.dump(solver, open("train_fcnet.dat", "wb"))  # HACK
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
