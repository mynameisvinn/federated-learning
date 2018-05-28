from sklearn.model_selection import train_test_split
import numpy as np

def fetch_minibatch(X, y, batch_size):
    n_points = X.shape[0]
    _, X_test, _, y_test = train_test_split(X, y, 
                                            test_size=float(batch_size)/n_points)
    return (X_test, y_test)