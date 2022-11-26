from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.multiply.outer(x_i, x_j)+1)**d

@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma*np.subtract.outer(x_i, x_j)**2)

@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x)
    K = kernel_function(x, x, kernel_param)
    return np.linalg.solve(K + _lambda * np.eye(n), y)

def predict(x_train, x_test, weight, kernel_function, kernel_param):
    return weight @ kernel_function(x_train, x_test, kernel_param)

def loss(x_train, x_test, y_test, weight, kernel_function, kernel_param):
    y_hat = predict(x_train, x_test, weight, kernel_function, kernel_param)
    return np.linalg.norm(y_test - y_hat) ** 2

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    n = len(x)
    fold_size = len(x) // num_folds
    to_delete = range(fold_size*num_folds, n)
    train_data = np.delete(np.c_[x, y], to_delete, 0)
    np.random.shuffle(train_data)
    sum_loss = 0.0
    for i in range(num_folds):
        train_fold = np.delete(train_data, range(i*fold_size, (i+1)*fold_size), 0)
        test_fold  = train_data[range(i*fold_size, (i+1)*fold_size)]
        weight = train(train_fold[:,0], train_fold[:,1], kernel_function, kernel_param, _lambda)
        sum_loss += loss(train_fold[:,0], test_fold[:,0], test_fold[:,1], weight, kernel_function, kernel_param)
    return sum_loss/num_folds
        


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    gamma_ = 1/np.median(abs(np.subtract.outer(x,x))) 
    best_lambda = 0.0
    best_gamma = gamma_
    best_loss = 99999999.9
    for i in np.linspace(-7,-1):
        for j in np.linspace(-1,2):
            _lambda = 10**i
            gamma = gamma_ * 10**j
            l = cross_validation(x,y,rbf_kernel,gamma,_lambda,num_folds)
            if l < best_loss:
                best_lambda = _lambda
                best_gamma = gamma
                best_loss = l
    return best_lambda, best_gamma


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    best_lambda = 0.0
    best_d = 1
    best_loss = 99999999.9
    for i in np.linspace(-7,-1):
        for d in range(7,22):
            _lambda = 10**i
            l = cross_validation(x,y,poly_kernel,d,_lambda,num_folds)
            if l < best_loss:
                best_lambda = _lambda
                best_d = d
                best_loss = l
    return best_lambda, best_d


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    # part a
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    x_train, y_train = x_300, y_300
    # num_folds = 10
    # _lambda_rbf, gamma = rbf_param_search(x_train, y_train, num_folds)
    # _lambda_poly, d = poly_param_search(x_train, y_train, num_folds)
    # print(f"rbf params: \n  lambda: {_lambda_rbf}, gamma: {gamma}")
    # print(f"poly params: \n  lambda: {_lambda_poly}, d: {d}")
    
    # x_30:
    # _lambda_rbf, gamma = 0.0019306977288832496, 40.39095772152451
    # _lambda_poly, d = 1.2648552168552959e-06, 17

    # x_300:
    _lambda_rbf = 0.04291934260128778
    gamma = 311.51859223228763
    _lambda_poly = 1e-07
    d = 21

    # part b
    weight_rbf = train(x_train, y_train, rbf_kernel, gamma, _lambda_rbf)
    weight_poly = train(x_train, y_train, poly_kernel, d, _lambda_poly)
    x_fine_grid = np.linspace(0, 1, 100)
    rbf_predictions = [predict(x_train, x, weight_rbf, rbf_kernel, gamma) for x in x_fine_grid]
    poly_predictions = [predict(x_train, x, weight_poly, poly_kernel, d) for x in x_fine_grid]
    # plt.plot(x_fine_grid, rbf_predictions, label = "rbf kernel")
    plt.plot(x_fine_grid, poly_predictions, label = "polynomial kernel")
    plt.plot(x_fine_grid, [f_true(x) for x in x_fine_grid], label = "true function")
    plt.legend()
    plt.show()
    
    


if __name__ == "__main__":
    main()
