#!/usr/bin/env python
# coding: utf-8

# # Task-1: Predict the percentage of an student based on the no. of study hours
# 
# # Author : Pinal Patel
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ### Step 1 : Import the libraries 

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# ###  Step 2 : Import the data

# In[2]:


df = pd.read_csv("http://bit.ly/w-data")
df


# ### Step 3 : Basic data information

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# ### Step 4 : Data Visualizations

# In[8]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Scores')  
plt.xlabel('Hours')  
plt.ylabel('Scores')  
plt.show()


# ### Step 5 : Splitting the data into train and test data

# In[17]:


X = df['Hours'].values
Y = df['Scores'].values


# In[18]:


X.shape


# In[19]:


Y.shape


# In[20]:


#Reshaping the data


X = df['Hours'].values.reshape(-1,1)
Y = df['Scores'].values.reshape(-1,1)


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[22]:


# Training the model

regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 


# In[23]:


# plotting the regression line

line = regressor.coef_*X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# In[24]:


# Testing data - In Hours
print(X_test)

# Predicting the scores
Y_pred = regressor.predict(X_test)


# In[28]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})
df


# In[30]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  


# In[ ]:




