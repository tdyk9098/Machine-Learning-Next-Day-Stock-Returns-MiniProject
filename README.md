# Machine-Learning-Next-Day-Stock-Returns-MiniProject
Developing models to predict continuous and categorical outputs for next day stock returns based off price data and investor sentiment data.

Dataset: Yahoo Finance (SPY, LQD), AAII Investor Sentiment Survey

36 Machine Learning models/algorithms were developed to study the behavior of SPY 1D returns AFTER the AAII Investor Sentiment Survey gets released.
Logistic Regression, Linear Regression, KNN, Decision Tree and Random Forest, Support Vector Machine models were run to predict categorical (up or down) and continuous next day returns.

Returns were defined two ways:
- (Close - Open) / Open of the following day :: All models with a 1 in their name are prediting this return.
- Modified Return: ((High - Open)/(Open-Low))/Open of the following day :: All models with a 2 in their name are predicting this return.

Model inputs were varied 4 times:
- Part 1 = SPY Price derivatives and Bull %, Bear %, Neutral % (and related derivatives)


# Observations: If Categorical Accuracy, Precision (Down, Up), Recall (Down, Up),and F-1 Score (Down, Up). If Continuous RMSE, RRMSE

No models do a sufficient enough job of predicting next day returns following the sentiment survey release. I believe this concludes the data used as inputs for the models ARE NOT predictive in nature to next day returns of the SPY ETF. Since the classes are quite balanced, we can use the accuracy to quickly compare all models.
By doing so, Part 1 KNN1 and Part 2 DT2 stand out. Nevertheless, the f-1 score remains imbalanced in favor of up days for KNN1 and down days for DT2. This leads me to the conclusion that neither models 
perform accurately enough to predict next day returns.

Top Models
- Part 1: KNN1


Part 1:
- LOGRM1 .53, (.54,.53), (.24,.80),(.33,.64)
- LOGRM2 .51,(.52,.48), (.82,.18), (.63,.26)
- LRM1 .009, 21.07
- LRM2 .011,11.00
- KNN1, adjusted with K = 41, .58, (.60,.57), (.34,.79), (.44,.64)
- KNN2, adjusted with K = 16, .51, (.53,.45), (.67,.32), (.59,.38)
- DT1 .50, (.47,.53), (.50,.49), (.49,.51)... RF1 .54, (.52,.56), (.5,.58), (.51,.57)
- DT2 .51, (.51,.51), (.54,.49), (.53,.50)... RF2 .48, (.49,.46), (.75,.21), (.59,.29)
- SM1 .52, (.53,.52), (.17,.86), (.25,.65)... Grid Search .51, (.50,.51), (.23,.78), (.32, .62)

# Install
pip install numpy | conda install numpy

pip install pandas | conda install pandas

pip install matplotlib | conda install matplotlib

pip install seaborn | conda install seaborn

pip install plotly

pip install cufflinks

pip install chart-studio

pip install -U scikit-learn

pip install scipy | conda install -c anaconda scipy

# Import
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

init_notebook_mode(connected = True)

import chart_studio.plotly as py

cf.go_offline()

%matplotlib inline

import pandas_datareader.data as web

import datetime as dt

import scipy.stats as st

from sklearn.metrics import r2_score 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
