# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = np.random.RandomState(seed=446)

class Linear(Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        alpha = 1/math.sqrt(d_in)
        dist = Uniform(-alpha, alpha)
        self.W = Parameter(dist.sample([d_out, d_in]))
        self.b = Parameter(dist.sample([d_out, 1]))
    def forward(self, x: torch.Tensor):
        return torch.mm(self.W, x) + self.b

class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.layer1 = Linear(d,h)
        self.layer2 = Linear(h,k)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        return self.layer2(relu(self.layer1(x.T))).T


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.layer1 = Linear(d, h0)
        self.layer2 = Linear(h0, h1)
        self.layer3 = Linear(h1, k)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        return self.layer3(relu(self.layer2(relu(self.layer1(x.T))))).T



@problem.tag("hw3-A")
def train(model: Module, x_train, y_train) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    optimizer = Adam(model.parameters())
    n, d = x_train.shape
    batch_size = 600
    num_batches = n//batch_size
    epochs = 1
    losses = []
    data = np.c_[x_train, y_train]
    for epoch in range(epochs):
        RNG.shuffle(data)
        batches = np.split(data, num_batches)
        for batch in batches:
            x, y = torch.from_numpy(batch[:, :d]).float(), torch.from_numpy(batch[:, d]).long()
            y_hat = model(x)
            loss = cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x, y = torch.from_numpy(data[:, :d]).float(), torch.from_numpy(data[:, d]).long()
        losses.append(cross_entropy(x, y))
    return losses

def accuracy(y_hat, y):
    n = y.shape
    num_incorrect = np.count_nonzero(np.argmax(y_hat, 1) - y )
    return 1. - num_incorrect/n
        

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    n, d = x.shape
    k = 10
    F1_model = F1(64, d, k)
    F2_model = F2(32, 32, d, k)
    F1_losses = train(F1_model, x, y)
    F2_losses = train(F2_model, x, y)

    print(f"F1 accuracy: {accuracy(F1_model(x_test), y_test)}")
    print(f"F2 accuracy: {accuracy(F2_model(x_test), y_test)}")

    plt.plot(range(50), F1_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(range(50), F2_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()
