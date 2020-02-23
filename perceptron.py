import numpy as np
import matplotlib.pyplot as plt
import json


def predict(coefs, x):
    """ requires coefs.shape == x.shape
    with or without intercept (bias) """
    print(f"x: {x}")
    print(f"bias {coefs[0]}, weights {coefs[1:]}")
    return coefs@x.T


def insert_intercept(X):
    """
    Inserts intercept (bias) to 0 index for 1d or 2d tensors (vector or matrix)
    For tensors of higher dimensions doesnt do anything
    """
    if len(X.shape) == 1:  # vector
        X = np.insert(X, 0, [1])
    elif len(X.shape) == 2:  # matrix
        bias_vec = np.ones(X.shape[0])
        X = np.insert(X, 0, bias_vec, axis=1)  # X with bias terms (1's) in first index
    return X


def fit(X, y, epochs=1):
    """ In progress
    args: X, y
    kwargs: epochs
    """
    # 1. initialize coef-vector
    # 2. iterate weight updates over # epochs
    coefs = np.random.rand(X.shape[1])
    print(f"[before] coefficients: {coefs}\n")
    print("-"*80)

    for i in range(epochs):
        # Perceptron algorithm updates weights after each misclassified sample (online learning).
        # Stream over each sample and perform update if sample is misclassified.
        # The size of update in Perceptron is always the size of sample values
        corrects = 0
        for j, x in enumerate(X):  # quite implicit, but iterates over samples
            y_hat = predict(coefs, x)
            print(f"x: {x} | y_hat: {y_hat}")
            # Update coefs if misclassified
            if (y_hat < 0 and y[j]==1):     # predicted=-1 and label=1 -> Misclassified
                coefs = coefs + x
            elif (y_hat >= 0 and y[j]==-1):  # predicted=1 and label=-1 -> Misclassified
                coefs = coefs - x
            else:  # remove from final version, just to count how many gets correct
                corrects += 1

            print(f"[after one update] coefficients: {coefs}\n")

        print("*"*60, "end of epoch")

        if corrects == X.shape[0]:
            print("All correct!")
            return coefs

    return coefs


def decision_function2d(x1, coefs):
    """
    Returns second coordinate range for drawing hyperplane in a 2 dimensional space.

    Args:
    - x1: first coordinate range
    - coefs: coefs[0]=intercept, coefs[1]=x1_weight, coefs[2]=x2_weight

    Solving the formula for decision function:
    => y = c0 + c1x1 + c2x2
    => 0 = c0 + c1x1 + c2x2  # 0 is the classification threshold in our step function
    => -c2x2 = c0 + c1x1
    => -x2 = (c0 + c1x1) / c2
    => x2 = -(c0 + c1x1) / c2
    """
    return (-coefs[0] - coefs[1]*x1)/coefs[2]


def plot_graph(X, y, xx, yy):
    def get_colors(y):
        colors = [None, 'green', 'red']  # class 1: green, class -1: red
        return [colors[c] for c in y]

    plt.scatter(X[:,0], X[:,1], c=get_colors(y))
    plt.plot(xx, yy)  # decision boundary
    plt.show()


if __name__ == "__main__":
    d = json.load(open("./test_data.json", "r"))

    case = d["or"]
    X = np.array(case["X"])
    y = np.array(case["y"])
    # 1. concatenate bias vector shape (1, X.nrows) into X
    #print(X)
    coefs = fit(insert_intercept(X), y, epochs=15)  # including intercept @ idx 0
    #print(X)
    print("Finally predicting:", predict(insert_intercept(X[0]), coefs))

    x1 = np.linspace(-0.5, 1.5, 10)
    x2 = decision_function2d(x1, coefs)

    plot_graph(X, y, x1, x2)

    """ for k, v in d.items():
        print(f"Fitting case {k}")
        c = fit(np.array(v["X"]), np.array(v["y"]), epochs=5)
        print("="*80, "\n") """
