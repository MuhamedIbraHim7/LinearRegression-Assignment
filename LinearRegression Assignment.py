#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Model 

# In[17]:


#Import Libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


# ## Data Reading 

# In[18]:


#load Salary Data
df = pd.read_csv('Salary_Data.csv')
df.head()


# In[19]:


#X Data
X = df.iloc[:,:-1].values
#y Data
y = df.iloc[:, 1].values


# In[20]:


# shape of Data
print(y.shape)
print(X.shape)


# In[21]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 33)


# In[22]:


# Fitting Simple Linear Regression to the Training set
LR = LinearRegression()
LR.fit(X_train, y_train)
print('Linear Regression Train Score is : ' , LR.score(X_train, y_train))
print('Linear Regression Test Score is : ' , LR.score(X_test, y_test))
print('Linear Regression Coef is : ' , LR.coef_)
print('Linear Regression intercept is : ' , LR.intercept_)


# ## Predictions

# In[23]:


#Calculating Prediction
y_pred = LR.predict(X_test)
print('Predicted Value for Linear Regression is : ' , y_pred[:10])


# ## Evaluation Process

# In[24]:


#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)


# In[25]:


#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)


# In[26]:


#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )


# In[27]:


print(X_test.shape, y_test.shape)
print(X_test.shape, y_pred.shape)


# ## visualisation

# In[28]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, LR.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[29]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, LR.predict(X_train), color = 'blue')
plt.title('Salary vs Experience ')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

