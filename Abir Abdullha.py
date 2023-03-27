#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd


# In[55]:


df = pd.read_excel(r'C:\Users\MSI\Downloads/Exam_Data.xlsx')


# In[56]:


df


# In[57]:


type(df.Load)


# In[58]:


df['load'] = pd.DataFrame(df.Load)
print (df.dtypes)


# In[59]:


df = pd.read_excel(r'C:\Users\MSI\Downloads/Exam_Data.xlsx')


# In[60]:


df['Load'] = df['Load'].astype(float)
print (df['Load'].dtypes)


# In[61]:


df['Load'].isna().sum()


# In[62]:


df


# In[63]:


df['Load'] = df['Load'].fillna(df['Load'].mean())


# In[64]:


df


# In[65]:


import matplotlib.pyplot as plt


# In[66]:


df['Load']=df['Load'].astype(float)


# In[67]:


df['Load'].plot(kind='hist', edgecolor='black')
print(df)


# In[68]:


df['Load_DAP'] = df['Load_DAP'].astype(float)
print (df['Load_DAP'].dtypes)


# In[69]:


df['Load_DAP'].isna().sum()


# In[70]:


df['Load_DAP'] = df['Load_DAP'].fillna(df['Load_DAP'].mean())


# In[71]:


df


# In[72]:


df. isnull().sum() 


# In[83]:


x=df.drop(['Previous_Point','Previous_Hour_Price','Sgn0_Volume_Dir','P24HA_Price','PDSH_Price','PWSH_Price','PWA_Price','Price_DAP'],axis=1)
x=x.dropna()
y = x['Target']
x = x.drop(['Target'],axis=1)


# In[85]:


x


# In[ ]:





# In[73]:


from sklearn.preprocessing import LabelEncoder


# In[86]:


from sklearn.preprocessing import LabelEncoder


# In[87]:


le =LabelEncoder()


# In[88]:


x['Date']= le.fit_transform(x['Date'])


# In[89]:


x['Date'].unique()


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[92]:


X_train.shape


# In[93]:


X_test.shape


# In[94]:


y_train.shape


# In[95]:


y_test.shape


# In[101]:


from sklearn.linear_model import LinearRegression


# In[102]:


logmodel=LinearRegression()


# In[103]:


logmodel.fit(X_train,y_train)


# In[105]:


x


# In[106]:


logmodel.predict([[22505,24,12,1,2,16798,8765,8766,8888]])


# In[108]:


logmodel.score(X_test,y_test)


# In[109]:


from sklearn.neighbors import KNeighborsRegressor


# In[110]:


log = KNeighborsRegressor(n_neighbors=2)


# In[112]:


log.fit(X_train, y_train)


# In[119]:


x


# In[124]:


log.predict([[22505,4,3,0,4,3440.496947,7427,7420.99,8427.874307]])


# In[125]:


log.score(X_test,y_test)


# In[129]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = [ 'Actual Target', 'Predicted Value']
students = [2938.9327,3365.35313333]
ax.bar(langs,students)
plt.title('Actual Target and Predicted Values')
plt.show()


# In[ ]:




