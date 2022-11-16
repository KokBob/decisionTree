# -*- coding: utf-8 -*-

# https://towardsdatascience.com/hyperparameters-of-decision-trees-explained-with-visualizations-1a6ef2f67edf
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y = True)
# %%
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
# %
import matplotlib.pyplot as plt
# %matplotlib inline
fig_size_tuple = (8,7)
plt.figure(figsize=fig_size_tuple)
tree.plot_tree(clf, filled=True, fontsize=8)
# %%
clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.2)
clf.fit(X, y)
plt.figure(figsize=fig_size_tuple)
tree.plot_tree(clf, filled=True, fontsize=14)
# %%
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
plt.figure(figsize=fig_size_tuple)
tree.plot_tree(clf, filled=True, fontsize=14)
# %%
clf = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=3)
clf.fit(X, y)
plt.figure(figsize=fig_size_tuple)
tree.plot_tree(clf, filled=True, fontsize=14)
# %%
clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf.fit(X, y)
plt.figure(figsize=fig_size_tuple)
tree.plot_tree(clf, filled=True, fontsize=14)
