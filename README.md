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
- Part 2 = LQD Price derivatives and Bull %, Bear %, Neutral % (and related derivatives)
- Part 3 = SPY and LQD Price derivatives and Bull %, Bear %, Neutral % (and related derivatives)
- Part 4 = Only Bull %, Bear %, Neutral % (and related derivatives)


# Observations: If Categorical Accuracy, Precision (Down, Up), Recall (Down, Up),and F-1 Score (Down, Up). If Continuous RMSE, RRMSE

No models do a sufficient enough job of predicting next day returns following the sentiment survey release. Since the classes are quite balanced, we can use the accuracy to quickly compare all models.
By doing so, Part 1 KNN1 and Part 2 DT2 stand out. Nevertheless, the f-1 score remains imbalanced in favor of up days for KNN1 and down days for DT2. This leads me to the conclusion that neither models 
perform accurately enough to predict next day returns.

Top Models
- Part 1: KNN1
- Part 2 : DT2


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
Part 2:
- LOGRM1 .54, (.50,.57), (.36,.70),(.42,.63)
- LOGRM2 .49,(.51,.43), (.80,.16), (.63,.24)
- LRM1 .01, 21.03
- LRM2 .009,35.69
- KNN1, adjusted with K = 40, .54, (.50,.56), (.34,.71), (.4,.63)
- KNN2, adjusted with K = 39, .50, (.51,.48), (.68,.3), (.58,.37)
- DT1 .47, (.48,.46), (.45,.48), (.47,.47)... RF1 .49, (.49,.48), (.34,.63), (.4,.55)
- DT2 .55, (.56,.53), (.66,.43), (.61,.48)... RF2 .45, (.48,.39), (.60,.28), (.53,.32)
- SM1 .57, (.53,.59), (.38,.72), (.44,.65)... Grid Search .55, (.50,.58), (.40,.68), (.44, .62)
Part 3:
- LOGRM1 .52, (.46,.56), (.38,.64),(.41,.6)
- LOGRM2 .45,(.5,.31), (.66,.19), (.57,.24)
- LRM1 .01, 56.70
- LRM2 .016,16.17
- KNN1, adjusted with K = 30, .56, (.57,.56), (.34,.76), (.43,.64)
- KNN2, adjusted with K = 7, .49, (.56,.43), (.45,.53), (.5,.47)
- DT1 .53, (.49,.56), (.47,.58), (.48,.57)... RF1 .54, (.51,.57), (.49,.6), (.50,.58)
- DT2 .53, (.59,.46), (.54,.51), (.57,.48)... RF2 .47, (.53,.4), (.51,.42), (.52,.41)
- SM1 .53, (.49,.54), (.17,.85), (.25,.66)... Grid Search .52, (.49,.55), (.48,.56), (.48, .55)
Part 4:
- LOGRM1 .56, (.49,.6), (.4,.68),(.44,.64)
- LOGRM2 .54,(.54,.53), (.92,.1), (.68,.17)
- LRM1 .008, 22.52
- LRM2 .01,15.63
- KNN1, adjusted with K = 39, .58, (.54,.6), (.43,.7), (.48,.65)
- KNN2, adjusted with K = 8, .52, (.56,.46), (.65,.37), (.60,.41)
- DT1 .51, (.51,.52), (.52,.51), (.51,.52)... RF1 .53, (.52,.53), (.45,.6), (.48,.56)
- DT2 .5, (.53,.46), (.54,.45), (.53,.46)... RF2 .53, (.55,.5), (.64,.42), (.59,.46)
- SM1 .55, (.58,.55), (.28,.81), (.38,.65)... Grid Search .54, (.53,.54), (.38,.69), (.44, .61)

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
