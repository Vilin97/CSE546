from tracemalloc import start
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2*np.sum(X*X, axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> np.ndarray:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Represents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    w = np.copy(weight)
    for (k, ak) in enumerate(a):
        ck = 2*(y - X @ w) @ X[:,k] + w[k]*ak
        if ck < -_lambda:
            w[k] = (ck + _lambda)/a[k]
        elif ck > _lambda:
            w[k] = (ck - _lambda)/a[k]
        else:
            w[k] = 0.0 
    return w

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.linalg.norm(y - X@weight,2) + _lambda * np.linalg.norm(weight,1)


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> np.ndarray:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        weight = np.zeros(X.shape[1])
    else:
        weight = start_weight
    a = precalculate_a(X)
    old_w = weight
    weight = step(X, y, weight, a, _lambda)
    while not convergence_criterion(weight, old_w, convergence_delta):
        old_w = weight
        weight = step(X, y, weight, a, _lambda)
        if loss(X,y,weight,_lambda) > loss(X,y,old_w,_lambda):
            print(loss(X,y,weight,_lambda), loss(X,y,old_w,_lambda))
    return weight


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    return max(abs(weight - old_w)) < convergence_delta


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    sigma = 1
    X = np.random.rand(n, d)
    w = np.concatenate( (np.arange(1,k+1)/k, np.zeros(d-k)) )
    y = X@w + np.random.normal(size=n)
    max_lambda = 2*max((y-np.average(y))@X)
    lambdas = [max_lambda*2**(-i) for i in range(10)]
    num_nonzero = np.zeros(len(lambdas)) 
    weight = None
    for (i,_lambda) in enumerate(lambdas):
        print(f"Done with {i} iterations")
        weight = train(X,y,_lambda,0.01)
        num_nonzero[i] = np.count_nonzero(weight)
    plt.plot(lambdas, num_nonzero)
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
