#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 State the Problem:  Determine what cutomers are most likelt to pay into their policies over time and make on time payments


# In[2]:


#2 Reading the file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[3]:


data = pd.read_csv('/Users/testuser/lab-customer-analysis-round-7/files_for_lab/csv_files/marketing_customer_analysis.csv')


# In[4]:


data.head()  #Showing the data in table form for python


# In[5]:


data.shape #understand how many rows and colums are before manipulation begins


# In[6]:


data.dtypes #understanding of numerical versus categorical values in the table


# In[7]:


data = data.drop (['Customer'], axis =1) #Removed because past anlysis shows it is not needed


# In[8]:


data = data.drop (['Effective To Date'], axis =1) #Removed because past anlysis shows it is not needed


# In[9]:


#3 Change Header Names
new_cols = []
for item in data.columns:
    new_cols.append(item.lower().replace(' ', '_'))


# In[10]:


new_cols


# In[11]:


data.columns = new_cols


# In[12]:


data.head()  


# In[ ]:





# In[ ]:





# In[13]:


#3.3 and #3.4 Check the data type before you run the numerical and categorical, this is done after cleaning the data

numericals = data.select_dtypes(np.number)
categoricals = data.select_dtypes(np.object)


# In[14]:


#3.2 Look for Nan values in data
len(data[data['income'].isna()==True])


# In[15]:


#3.5 Uncertain what "exploration" is of data


# In[ ]:





# In[16]:


#4 Outliers
sns.displot(data['customer_lifetime_value']) # data shows right tail and this is where the outliers are


# In[17]:


for num_col in data.select_dtypes(np.number).columns:
    print(num_col)
    sns.distplot(data[num_col])
    plt.show()   # have tried to run multiple ways do not understand these pink box errors ;-/


# In[18]:


#4.2 Normalization of data
data.isna().sum()/len(data)*100  #demostrated from Python L2


# In[19]:


IQR = np.percentile(data['customer_lifetime_value'], 75)- np.percentile(data['customer_lifetime_value'], 25)
#box plot caluclation


# In[20]:


u_limit = np.percentile(data['customer_lifetime_value'], 75) + 5*IQR


# In[21]:


l_limit = np.percentile(data['customer_lifetime_value'], 25) - 1.5*IQR


# In[22]:


np.percentile(data['customer_lifetime_value'], 75)


# In[23]:


outliers = data[(data['customer_lifetime_value']>u_limit)] 
(data['customer_lifetime_value']<l_limit)


# In[24]:


outliers.shape #amount of outliers


# In[25]:


outliers.shape[0]/data.shape[0]*100  #this isa good value becuse it is less than 5%


# In[26]:


#to refine the data seen in plython lab 5
data = data[(data['customer_lifetime_value']<u_limit) & (data['customer_lifetime_value']>l_limit)]
data = data.reset_index(drop=True)


# In[27]:


#review chart of refined data
sns.distplot(data['customer_lifetime_value'])


# In[28]:


#6 R2 Transform data once refined
def sq_rt_transform(x):
    if x<=0:
        return
    else:
        return x**0.5


# In[29]:


temp = list(map(sq_rt_transform, data['customer_lifetime_value']))


# In[30]:


sns.distplot(temp)  #with refined data the plot is less skewed and is taking on more of a classic bellcurve


# In[31]:


#4.3 Encoding Categotical Data


# In[32]:


data.head()


# In[33]:


data_correlation = data.corr()


# In[34]:


data_correlation  # to see the heat map of data


# In[35]:


sns.heatmap(data_correlation, annot=True) # use the values that match at one as these are the best matches


# In[36]:


data.isna().sum()


# In[ ]:





# In[ ]:





# In[43]:


scaled = StandardScaler().fit_transform(numericals)


# In[44]:


scaled


# In[45]:


scaled = pd.DataFrame(scaled)


# In[46]:


scaled


# In[47]:


encoded = OneHotEncoder(drop='first').fit_transform(categoricals).toarray()  #why error


# In[48]:


categoricals.shape


# In[49]:


encoded


# In[50]:


encoded = pd.DataFrame(encoded)


# In[51]:


encoded


# In[52]:


#4.4 Splitting into Train and Test Set for Modeling


# In[55]:


numericals = x.select_dtypes('number')
categoricals = x.select_dtypes('object')


# In[42]:


y = data['total claim amount']
features = data.drop(['total claim amount'], axis =0)


# In[53]:


numericals = features.select_dtypes(np.number)
categoricals = features.select_dtypes(np.object)


# In[ ]:





# In[ ]:





# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size= 0.1, random_state=100)


# In[ ]:


#5 Modeling


# In[ ]:


x_train.shape


# In[ ]:


len(y_train)


# In[ ]:


x_test.shape


# In[ ]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[ ]:


predictions = lm.predict(x_test)


# In[ ]:


predictions[:10] # y hat, what is this?


# In[ ]:


y_test[:10]  #real Y


# In[ ]:


e1 = 312-362
e2 = 131-266
e3 = 65-39


# In[ ]:


e1


# In[ ]:


e2


# In[ ]:


e3


# In[ ]:


total error = e1**2 + e2**2 + e3**2 + ......e914**2


# In[ ]:


total error/914


# In[ ]:


#6.2 MSE
mse = mean_squared_error(predictions, y_test)


# In[ ]:


mse


# In[ ]:


#6.3 RMSE
rmse = mse**0.5
rmse


# In[ ]:


#6.4 MAE
import math
math.sqrt(mse)


# In[ ]:


mae - Mean_absolute_error(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




