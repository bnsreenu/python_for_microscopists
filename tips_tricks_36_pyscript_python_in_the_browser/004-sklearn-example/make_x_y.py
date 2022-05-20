# -*- coding: utf-8 -*-
"""

"""


from sklearn.datasets import make_blobs


def make_x_and_y():
    x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return x, y

