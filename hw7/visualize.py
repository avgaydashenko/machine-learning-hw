from matplotlib.mlab import find
import numpy as np
import matplotlib.pyplot as pl

def visualize(clf, X, y):
    border = .5
    h = .02

    x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
    y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    pl.figure(1, figsize=(8, 6))
    pl.pcolormesh(xx, yy, z_class, cmap=pl.cm.summer, alpha=0.3)
    
    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    pl.contour(xx, yy, z_dist, [0.0], colors='black')
    pl.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    y_pred = clf.predict(X)
    
    ind_correct = list(set(find(y == y_pred)))
    ind_incorrect = list(set(find(y != y_pred)))
    
    pl.scatter(X[ind_correct, 0], X[ind_correct, 1], c=y[ind_correct], cmap=pl.cm.summer, alpha=0.9)
    pl.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=y[ind_incorrect], cmap=pl.cm.summer, alpha=0.9, marker='*', s=50)
    
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())