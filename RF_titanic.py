#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd 
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as plt

from sklearn import preprocessing
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import tree

from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image  


# In[2]:


os.chdir('D:\\Downloads\\')


# In[3]:


os.getcwd()


# In[4]:


df = pd.read_csv(r'D:\Downloads\train.csv')
df1 = pd.read_csv(r'D:\Downloads\test.csv')


# In[5]:


df.head()


# In[6]:


df1.isnull().sum()


# In[7]:


df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'Embarked'], axis=1, inplace=True)
df1.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[8]:


combine = [df,df1]


# In[ ]:





# In[9]:


def change_sex(sex):
    if sex == "male":
        return 1
    elif sex == "female":
        return 0


# In[10]:


df["Sex"] = df.apply(lambda row : change_sex(row["Sex"]),axis = 1)


# In[11]:


df1["Sex"] = df1.apply(lambda row : change_sex(row["Sex"]),axis = 1)


# In[12]:


df1.head()


# In[13]:


import math
female_mean, male_mean = df.groupby("Sex")["Age"].mean()
def fill_age(age,sex):
    if math.isnan(age):
        if sex == 1:
            return male_mean
        else:
            return female_mean
    else:
        return age


# In[14]:


df["Age"] = df.apply(lambda row : fill_age(row["Age"],row["Sex"]),axis = 1)


# In[15]:


df1["Age"] = df1.apply(lambda row : fill_age(row["Age"],row["Sex"]),axis = 1)


# In[16]:


x = df.drop(['Survived'], axis =1)
y = df['Survived']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state = 42)


# In[18]:


#dtree = DecisionTreeClassifier()


# In[19]:


#dtree.fit(x_train, y_train)


# In[20]:


rf = RandomForestClassifier(random_state = 42)


# In[21]:


rf.fit(x_train, y_train)


# In[22]:


y_test_pred = rf.predict(x_test)


# In[23]:


accuracy_score(y_test, y_test_pred)


# In[24]:


y_train_pred = rf.predict(x_train)


# In[25]:


accuracy_score(y_train, y_train_pred)


# In[26]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2, 5, 10, 15, 20]

min_samples_leaf = [1, 2, 4, 6, 8]

bootstrap = [True, False]


# In[27]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[28]:


rf = RandomForestClassifier(random_state = 42)


# Randomized Search:

# In[29]:


rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100,  
                              cv = 5, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)


# In[30]:


rf_random.fit(x_train, y_train)


# In[31]:


best_params=rf_random.best_params_


# In[32]:


best_params


# In[33]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, x_train, y_train, x_test, y_test, train=True):
    if train:
        y_train_pred = clf.predict(x_train)
        clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, y_train_pred)}\n")
        
    elif train==False:
        y_test_pred = clf.predict(x_test)
        clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_test_pred)}\n")


# In[ ]:





# In[34]:


tree_clf = RandomForestClassifier(**best_params)
tree_clf.fit(x_train, y_train)


# In[35]:


print_score(tree_clf, x_train, y_train, x_test, y_test, train=True)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)


# In[36]:


'''def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    '''


# In[37]:


'''best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)'''


# In[38]:


#rf_random.cv_results_


# In[39]:


#rf = RandomForestClassifier(n_estimators= 800,
#max_depth=90, 
#max_features='sqrt', 
#min_samples_split = 2,
#min_samples_leaf = 4,                            
#bootstrap = True)


# Custom:

# In[40]:


rf = RandomForestClassifier(n_estimators= 600,
 min_samples_split = 15,
 min_samples_leaf= 25,
 max_features= 'auto',
 max_depth= 60,
 bootstrap= False)


# In[41]:


rf.fit(x_train,y_train)


# In[42]:


y_pred_train = rf.predict(x_train)
y_pred = rf.predict(x_test)


# In[43]:


print("Training accuracy: ",accuracy_score(y_train,y_pred_train))
print("Test accuracy: ",accuracy_score(y_test,y_pred))


# Grid Search:

# In[54]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    #'max_depth': [5, 8, 15, 25, 30],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5, 10],
    'min_samples_split': [8, 10, 12, 14],
    #'min_samples_split' : [2, 5, 10, 15, 100],
    'n_estimators': [100, 200, 300,500, 1000]
}


# In[55]:


rf = RandomForestRegressor(random_state = 42)


# In[56]:


grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)


# In[57]:


grid_search.fit(x_train, y_train);


# In[58]:


best_params=grid_search.best_params_


# In[59]:


best_params


# In[60]:


rf = RandomForestClassifier(**best_params)
rf.fit(x_train, y_train)


# In[61]:


print_score(rf, x_train, y_train, x_test, y_test, train=True)
print_score(rf, x_train, y_train, x_test, y_test, train=False)


# In[52]:


#rff = RandomForestClassifier(n_estimators=500,criterion='gini', max_depth=80, max_features=3, min_samples_leaf=5,
#                       min_samples_split=20, bootstrap=True)
#rff.fit(x_train, y_train)


# In[ ]:





# In[ ]:




