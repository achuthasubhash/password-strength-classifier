# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:19:35 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt 
 
data=pd.read_csv("E:\\github\\my upload\\password strength\\data.csv",',',error_bad_lines=False)

data.head()

data['password'].unique()

data.isnull().any()
#as password has null so delete it with out replacement
data.dropna(inplace=True)

data.isnull().any()
 
data.shape

#plotting checking balance or not
count = pd.value_counts(data['strength'], sort = True)
count.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

#into characetrs
def character(input):
    char=[]
    for i in input:
        char.append(i)
    return char

X=character(data['password'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=character)
X=vectorizer.fit_transform(X)

X.shape

vectorizer.vocabulary_

y=data['strength']


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))

# Creating a pickle file for the CountVectorizer
pickle.dump(vectorizer, open('vectorizer-transform.pkl', 'wb'))
# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'password strength.pkl'
pickle.dump(classifier, open(filename, 'wb'))
