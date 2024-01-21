#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Loan Approval Prediction Project


# In[100]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


df=pd.read_excel("C:\\Users\\MonishaAnthony\\Downloads\\Copy of loan.xlsx")


# In[102]:


df.head()


# In[103]:


df.info()


# In[104]:


df.isnull().sum()


# In[ ]:





# In[105]:


df.isnull().sum()


# In[ ]:





# In[106]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())


# In[107]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

df.isnull().sum()


# In[108]:


df=df.drop('Loan_ID', axis=1)


# In[109]:


df.tail()


# In[110]:


df['Dependents']= df['Dependents'].replace(to_replace="3+", value = "4")


# In[111]:


df.tail()


# In[112]:


df['Dependents'].unique()


# In[113]:


df['Gender'].unique()


# In[114]:


df['Gender']=df['Gender'].map({'Male':1, 'Female':0}).astype('int')
df['Married']=df['Married'].map({'No':1, 'Yes':0}).astype('int')
df['Education']=df['Education'].map({'Graduate':1, 'Not Graduate':0}).astype('int')
df['Self_Employed']=df['Self_Employed'].map({'No':1, 'Yes':0}).astype('int')
df['Property_Area']=df['Property_Area'].map({'Urban':1, 'Rural':0, 'Semiurban':0}).astype('int')
df['Loan_Status']=df['Loan_Status'].map({'Y':1, 'N':0}).astype('int')


# In[115]:


df['Gender'].unique()


# In[116]:


df['Married'].unique()


# In[117]:


df['Education'].unique()


# In[118]:


df['Self_Employed'].unique()


# In[119]:


df['Property_Area'].unique()


# In[120]:


df['Loan_Status'].unique()


# In[121]:


df.head()


# In[122]:


##Storing the dependent and independent features


# In[123]:


X = df.drop('Loan_Status', axis =1)


# In[124]:


X


# In[125]:


Y = df['Loan_Status']


# In[126]:


Y


# In[127]:


## scaling


# In[128]:


cols = ['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[129]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X[cols]= ss.fit_transform(X[cols])


# In[130]:


X


# In[131]:


print("Dependent status of the people")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df, palette='Set1')


# In[132]:


print("Employment status of the people")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df, palette='Set1')


# In[133]:


print("Number of people who take loan group as Credit history")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df, palette='Set1')


# In[134]:


## Training and testing data set


# In[135]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[152]:


mod_df={}
def model_val(model, X,Y) :
    X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    model.fit(X_train,Y_train)
    y_Pred=model.predict(x_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_Pred)}")
    
    score=cross_val_score(model, X,Y, cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    mod_df[model]=round(np.mean(score)*100,2)


# In[153]:


#Logistic Regression


# In[154]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,Y)


# In[140]:


mod_df


# In[141]:


##Support Vector Classifier


# In[142]:


from sklearn import svm
model=svm.SVC()
model_val(model,X,Y)


# In[143]:


mod_df


# In[144]:


## Decision tree clasifier


# In[145]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,Y)


# In[146]:


mod_df


# In[ ]:


##Random Forest Classifier


# In[147]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model_val(model,X,Y)


# In[148]:


mod_df


# In[149]:


##Gradient Boosting classifier


# In[150]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model_val(model,X,Y)


# In[151]:


mod_df


# In[ ]:


##Hyperparameter tuning


# In[155]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


##Logistic Regression


# In[161]:


log_reg_grid={"C":np.logspace(-4,4,20),"solver":['liblinear']}


# In[162]:


rs_log_reg = RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,n_iter=20,cv=5,verbose=True)


# In[163]:


rs_log_reg.fit(X,Y)


# In[165]:


rs_log_reg.best_score_


# In[166]:


rs_log_reg.best_params_


# In[ ]:


##Support Vector Classifier


# In[173]:


sv_grid = {'C':[0.13,3,0.47,1.4],'kernel':["linear"]}


# In[177]:


rs_svc = RandomizedSearchCV(svm.SVC(),param_distributions=sv_grid,cv=5,n_iter=20,verbose=True)


# In[178]:


rs_svc.fit(X,Y)


# In[179]:


rs_svc.best_score_


# In[180]:


rs_svc.best_params_


# In[ ]:


##Random Forest Classifier


# In[187]:


RandomForestClassifier()


# In[188]:


rf_grid ={'n_estimators':np.arange(10,1000,10),'max_features' : ['auto','sqrt'],'max_depth':[None,3,5,8,24,43],'min_samples_leaf':[6,9,10,12],
'min_samples_leaf':[5,2,7,10]}


# In[189]:


rs_rf = RandomizedSearchCV(RandomForestClassifier(),param_distributions=rf_grid,cv=5,n_iter=20,verbose=True)


# In[190]:


rs_rf.fit(X,Y)


# In[191]:


rs_rf.best_score_


# In[192]:


rs_rf.best_params_


# In[ ]:


Logistic Regression before Hyperparameter tuning is 80.46
Logistic Regression after Hyperparameter tuning is 80.62

SVC score before hyperparameter tuning is 79.64
SVC score after hyperparameter tuning is 80.94

RandomForest classifier before hyperparameter tuning is 78.83
RandomForest classifier before hyperparameter tuning is 80.94


# In[ ]:


## Training the best model with best parameters for whole dataset (Random forest classifier)


# In[ ]:


X = df.drop('Loan_Status', axis =1)
Y = df['Loan_Status']


# In[193]:


rf= RandomForestClassifier(n_estimators = 60,
 min_samples_leaf = 2,
 max_features ='sqrt',
 max_depth = 5)


# In[194]:


rf.fit(X,Y)


# In[195]:


import joblib


# In[196]:


joblib.dump(rf,'Loan_status_pred')


# In[197]:


model = joblib.load('Loan_status_pred')


# In[ ]:





# In[ ]:


##checking with new sample data with the saved model


# In[207]:


import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':28890,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':1,
    'Property_Area':1
},index=[0])


# In[208]:


df


# In[209]:


Status = model.predict(df)


# In[210]:


if Status==1:
    print("Loan is Approved")
else:
    print("Loan is not Approved")


# In[ ]:





# In[ ]:





# In[ ]:




