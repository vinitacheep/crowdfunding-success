#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[ ]:


import numpy as np
import pandas as pd
import os
import random
import time
import matplotlib.pyplot as plt
from collections import Counter


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score

import tensorflow as tf


# In[ ]:


#import data
df = pd.read_csv("/Users/advait/Downloads/encodings (1).csv")

#split data into train/test
X = df

df_outcomes = pd.read_csv("/Users/advait/Downloads/small_filtered_geotagged_surgery_campaigns.csv")
y_new = df_outcomes['raised_amnt'].to_numpy() / df_outcomes['goal_amnt'].to_numpy()

y = []
for item in y_new:
    if item >= 1:
        y.append(1)
    else:
        y.append(0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=5)


# In[ ]:


param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000]
        }

log = LogisticRegression(penalty='l1', solver='liblinear')
cv = StratifiedKFold(5, shuffle=True, random_state=5)
model = GridSearchCV(log, param_grid, cv = cv, refit = True, verbose=True, n_jobs = -1, scoring = 'average_precision')

get_ipython().run_line_magic('time', 'model.fit(x_train, y_train)')


# In[ ]:


best_params = model.best_estimator_.get_params()
best_params


# In[ ]:


#assess performance

y_pred = model.predict_proba(x_test)
# Convert proabilities to binary output for performance assessment, 0.5 cutoff
y_pred_binary = np.where(y_pred[:,1] > 0.5, 1, 0)

# Metrics at different probability cutoffs
# (pos label assigned if pos class probability > cutoff)

cutoffs = np.arange(0.1, 1, 0.1)
target_names = ['negative', 'positive']

for cutoff in cutoffs:
  print('Cutoff: ', round(cutoff, 2))
  print(classification_report(y_test, np.where(y_pred[:,1] > cutoff, 1, 0), target_names=target_names))


# In[ ]:


# Metrics at Positive class Recall of at least 90

from sklearn.metrics import f1_score
# get roc
y_pos = y_pred[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pos)

# get index of sensitivity closest to 90+
indeces = np.where(tpr >= 0.9)[0]
i = indeces[tpr[indeces].argmin()]

# calculate metrics
auroc = auc(fpr, tpr)
sens = tpr[i]*100
spec = (1 - fpr[i])*100
cutoff = thresholds[i]
y_pred_bin = (y_pred[:,1] >= cutoff).astype(int)
acc = accuracy_score(y_test, y_pred_bin)*100
avg_prec = average_precision_score(y_test, y_pos)*100
f1 = f1_score(y_test, y_pred_bin)*100

print("Recall:\t\t\t", round(sens, 3))
print("Specificity:\t\t", round(spec,3))
print("AUROC:\t\t\t", round(auroc, 3))
print("Probabitiy cutoff:\t", round(cutoff, 3))
print('Accuracy:\t\t', round(acc, 3))
print('Average Precision:\t', round(avg_prec, 3))
print('F1-score:\t\t', round(f1, 3))


# In[ ]:


# plot ROC curve
plt.plot(fpr, tpr, label='Linear Classifier (AUROC = %0.2f)' % (auroc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.axhline(y=sens/100, color='r', linestyle='--', linewidth=1)
plt.axvline(x=1-spec/100, color='r', linestyle='--', linewidth=1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate (Recall)')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#get coefficients and feature names
param_list = model.best_estimator_.coef_
param_list_new = []
for item in param_list:
    for element in item:
        param_list_new.append(element)

cols = list(X.columns)

#get total initial features and non-zero featuers after CV
print(len(param_list[0]))
np.count_nonzero(param_list)


# ## T-SNE and PCA Plot Generation

# In[ ]:


from sklearn.manifold import TSNE
from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("/Users/advait/Downloads/encodings (1).csv")
df


# In[ ]:


#bin crowdfunding campaign successes

df_outcomes = pd.read_csv("/Users/advait/Downloads/small_filtered_geotagged_surgery_campaigns.csv")
df_outcomes['percent_raised'] = df_outcomes['raised_amnt'].to_numpy() / df_outcomes['goal_amnt'].to_numpy()

y = pd.qcut(df_outcomes['percent_raised'], 4, labels=False)
y = y+1
y


# In[ ]:


tsne = TSNE(n_components=2, verbose=1)
x = df.to_numpy()
z = tsne.fit_transform(x) 


df = pd.DataFrame()
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
df['y'] = y

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Crowdfunding data T-SNE projection") 


# In[ ]:


pca = PCA(n_components=2)
X_r = pca.fit(x).transform(x)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], y):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of Crowdfunding Campaigns")


# In[ ]:





# In[ ]:




