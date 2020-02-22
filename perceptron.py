import numpy as np
import json


def predict(x, coefs):
    print(f"bias {coefs[0]}, weights {coefs[1:]}")


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
    print(X)
    coefs = np.random.rand(X.shape[1])
    print(f"[before] coefficients: {coefs}\n")
    print("-"*80)

    for i in range(epochs):
        # Perceptron algorithm updates weights after each misclassified sample (online learning).
        # Stream over each sample and perform update if sample is misclassified.
        # The size of update in Perceptron is always the size of sample values
        corrects = 0
        for j, x in enumerate(X):  # quite implicit, but iterates over samples
            y_hat = coefs@x.T
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


if __name__ == "__main__":
    d = json.load(open("./test_data.json", "r"))

    case = d["and"]
    X = np.array(case["X"])
    y = np.array(case["y"])
    coefs = fit(X, y, epochs=10000)  # including intercept @ idx 0
    predict(X[0], coefs)
    """ for k, v in d.items():
        print(f"Fitting case {k}")
        c = fit(np.array(v["X"]), np.array(v["y"]), epochs=5)
        print("="*80, "\n") """
