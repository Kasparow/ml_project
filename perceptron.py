import numpy as np


X1 = np.array([
    [2, 5],
    [4, 1],
    [5, 5],
    [1, 3]
])

X = np.array([
    [2],
    [4],
    [5],
    [1]
])

y = np.array([-1, 1, 1, -1])


def fit(X, y, epochs=1):
    """ In progress
    args: X, y
    kwargs: epochs
    """
    # 1. concatenate bias vector shape (1, X.nrows) into X
    # 2. initialize coef-vector
    # 3. iterate weight updates over # epochs
    bias_vec = np.ones(X.shape[0])
    X = np.insert(X, 0, bias_vec, axis=1)  # X with bias terms (1's) in first index

    coefs = np.random.rand(X.shape[1])
    print("Before\n", coefs)
    print("-"*80)

    for i in range(epochs):
        # Perceptron algorithm updates weights after each misclassified sample (online learning).
        # Stream over each sample and perform update if sample is misclassified.
        # The size of update in Perceptron is always the size of sample values
        for j, x in enumerate(X):  # quite implicit, but iterates over samples
            y_hat = coefs@x.T
            print("x and y_hat", x, y_hat)
            # Update coefs if misclassified
            if (y_hat < 0 and y[j]==1):     # predicted=-1 and label=1 -> Misclassified
                coefs = coefs + x
            elif (y_hat >= 0 and y[j]==-1):  # predicted=1 and label=-1 -> Misclassified
                coefs = coefs - x

            print("coefs:", coefs)
        print("*"*60, "end of epoch")


if __name__ == "__main__":
    fit(X, y, epochs=5)
