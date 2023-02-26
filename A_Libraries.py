##############################################################################
#                                                                            #
#                            A. Import Libraries                            #
#                                                                            #
##############################################################################

def Libraries():
    import os
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from sklearn.cluster import KMeans
    from mpl_toolkits import mplot3d
    import seaborn as sn
    from numpy import matlib
    import itertools
    from scipy import linalg
    import matplotlib as mpl
    from sklearn import mixture
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    import warnings
    return os, pd, StandardScaler, sc, PCA, np, plt, KMeans, mplot3d, sn, rc, matlib, itertools, linalg, mpl, mixture, DBSCAN, metrics, make_blobs, warnings

