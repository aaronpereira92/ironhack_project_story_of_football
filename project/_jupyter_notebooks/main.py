#!/usr/bin/env python
# coding: utf-8

# Based on the modelling notebook from before, here I can import the functions, tansformers and models that I defined in the modelling notebook, and apply it to any new dataset that will be given to me in the future.
# 
# As I currently, have no new dataset, and for purposes of testing my functions, transformers and models out, I will import the **X_test**, **y_test** csv files

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import joblib
import pickle 


# In[2]:


#reading in the files for using my model with
#generally when new data comes in, we won't have the y variable with it, so theres no need to read the y variable in
#But just doing it here for the sake of testing out my functions, transfomers and models
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')


# In[3]:


#Before we can model, we need to Transform our data
#First step is split into categoricals and numericals
# From my functions I do so using the num_cat_splitter 
#This function returns 3 dataframes - numerical, categorical ordinals and categorical nominal
from functions import num_cat_splitter


# In[4]:


num_test,cat_ord_test, cat_nom_test = num_cat_splitter(X_test)


# In[5]:


display(num_test.head())
display(cat_ord_test.head())
display(cat_nom_test.head())


# In[6]:


#Now comes the transformations
filename = 'minmaxscaler_numericals.sav'
minmax = pickle.load(open(filename, 'rb'))
num_test_norm = minmax.transform(num_test)
num_test_trans = pd.DataFrame(num_test_norm, columns=num_test.columns)


# In[7]:


num_test_trans


# In[8]:


filename = 'onehotencoder_catnom.sav'
ohe = pickle.load(open(filename, 'rb'))
cat_nom_test_ohe = ohe.transform(cat_nom_test).toarray()
cols = ohe.get_feature_names(input_features=cat_nom_test.columns)
cat_nom_test_encoded = pd.DataFrame(cat_nom_test_ohe, columns=cols)


# In[9]:


cat_nom_test_encoded


# In[10]:


filename = 'labelencoder_catord.sav'
label_encoder = pickle.load(open(filename, 'rb'))
cat_ord_test_le = label_encoder.transform(cat_ord_test)
cat_ord_test_encoded = pd.DataFrame(cat_ord_test_le, columns=cat_ord_test.columns)


# In[11]:


cat_ord_test_encoded


# In[12]:


X_test_trans = pd.concat([cat_nom_test_encoded,cat_ord_test_encoded,num_test_trans], axis = 1)


# In[13]:


X_test_trans


# In[14]:


#Now its time to import the model that was already fit with train SMOTE data in the previous notebook/
filename = 'football_logisticreg_smote_model.sav'
loaded_model = joblib.load(filename)


# In[17]:


#Here are the predictions!
pred = loaded_model.predict(X_test_trans)
pred


# In[21]:


#But for this purpose lets look back at our confusion matrix for the test set
# Test set confusion matrix
print("The confusion matrix on the TEST set is: ")
cm_test = confusion_matrix(y_test, pred)
cm_test
disp = ConfusionMatrixDisplay(cm_test,display_labels=loaded_model.classes_);
disp.plot()
plt.savefig('confusion_matrix_smote.png', bbox_inches='tight')
plt.show()


# In[19]:


#Here are the error metrics
print("The accuracy in the TEST  set is: {:.3f}".format(accuracy_score(y_test, pred)))
display(precision_score(y_test, pred, average = None))
display(recall_score(y_test,  pred, average = None))
f1_score(y_test, pred,average = None)
print("The cohen_kappa_score in the TEST  set is: {:.3f}".format(cohen_kappa_score(y_test, pred)))


# **The END**
