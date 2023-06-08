#!/usr/bin/env python
# coding: utf-8

# # Problem
# we will try to build a model that will predict the price of a car depending on its most important attributes. So the question is what factors or variables affect the price of a car.

# # Libraries import

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython import display



get_ipython().run_line_magic('matplotlib', 'inline')


# # DATA IMPORT

# In[2]:


data = pd.read_csv('CarPrice_Assignment.csv')
data.head()


# In[3]:


data.info()


# In[4]:


pd.isnull(data).any()


# In[5]:


pd.isna(data).any()


# In[6]:


data['CarName'].unique()


# In[7]:


data['model'] = [x.split()[0] for x in data['CarName']] # conversion to correct names
data['model'] = data['model'].replace({'maxda': 'Mazda','mazda': 'Mazda', 
                                     'nissan': 'Nissan', 
                                     'porcshce': 'Porsche','porsche':'Porsche', 
                                     'toyouta': 'Toyota', 'toyota':'Toyota',
                            'vokswagen': 'Volkswagen', 'vw': 'Volkswagen', 'volkswagen':'Volkswagen'})


# In[8]:


data= data.drop(['car_ID','CarName'], axis=1)


# # Data visualisation

# In[9]:


model_highwaympg = data.groupby('model')['highwaympg'].mean()
print(model_highwaympg)

plt.figure(figsize=(10, 6))
model_highwaympg.plot(kind='bar',color='#887672')

plt.title('Mean higway mpg by car model')
plt.xlabel('Model')
plt.ylabel('Mean higway mpg')
plt.xticks(rotation=45)
plt.tight_layout()


# In[10]:


model_citympg = data.groupby('model')['citympg'].mean()
print(model_citympg)

plt.figure(figsize=(10, 6))
model_citympg.plot(kind='bar',color='#887672')

plt.title('Mean city mpg by car model')
plt.xlabel('Model')
plt.ylabel('Mean city mpg')
plt.xticks(rotation=45)
plt.tight_layout()


# ### based on this we can calculate a mean mpg

# In[11]:


data = data.assign(mean_mpg=(data['citympg'] + data['highwaympg']) / 2)

mean_mpg_by_model = data.groupby('model')['mean_mpg'].mean()

plt.bar(mean_mpg_by_model.index, mean_mpg_by_model.values, color = '#1E4059')
plt.xticks(rotation=90)
plt.xlabel('Marka')
plt.ylabel('Mean mpg')
plt.title('Mean mpg by cars model')
plt.show()

data.drop('mean_mpg',axis=1 ,inplace=True)


# In[12]:


model_counts = data['model'].value_counts()
print(model_counts)

plt.pie(model_counts, labels=model_counts.index, autopct='%.1f%%')

plt.title('Count of Cars by Model')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# The largest collection in our data are Toyotas

# In[13]:


carbody_counts = data['carbody'].value_counts()
print(carbody_counts)

plt.pie(carbody_counts, labels=carbody_counts.index, autopct='%.1f%%')

plt.title('Count of Cars by Body Type')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# Sedans are the most popular.

# In[14]:


enginelocation_counts = data['enginelocation'].value_counts()
print(enginelocation_counts)

plt.pie(enginelocation_counts, labels=enginelocation_counts.index, autopct='%.1f%%')
plt.title('Engine Location Counts')
plt.show()


# most of the cars have engine located at the front. Rear-engined cars are usually more expensive.

# # Dummies

# In[15]:


lab=LabelEncoder()
data['fuelsystem']=lab.fit_transform(data['fuelsystem'])
data['cylindernumber']=lab.fit_transform(data['cylindernumber'])
data['enginetype']=lab.fit_transform(data['enginetype'])
data['enginelocation']=lab.fit_transform(data['enginelocation'])
data['drivewheel']=lab.fit_transform(data['drivewheel'])
data['carbody']=lab.fit_transform(data['carbody'])
data['doornumber']=lab.fit_transform(data['doornumber'])
data['aspiration']=lab.fit_transform(data['aspiration'])
data['fueltype']=lab.fit_transform(data['fueltype'])
data['model']=lab.fit_transform(data['model'])


# In[16]:


data.price.skew()


# 
# A positive skewness value (greater than zero) acts on the skewness to the right, which means that the tail of the distribution is to the right. This suggests that it may contain several values

# In[17]:


# price distribution
plt.figure(figsize=(10,6))
sns.distplot(data['price'], bins = 30, hist=True, kde=True, color='teal')

plt.xlabel('Price of cars [$]')
plt.ylabel('Count of cars')
plt.title('Count of cars depending of price [$]')
plt.show()


# In[18]:


#sns.pairplot(data)


# In[19]:


data.corr() #PEARSON CORR

mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,9))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={'size':10})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[20]:


data = data.drop('citympg',axis=1)


# # Model
# basic model without any changes

# In[21]:


prices = data['price']
features = data.drop('price', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])
print(coefficients)


# In[22]:


data['price'].skew()


# In[23]:


y_log = np.log(data['price'])
y_log.tail()


# In[24]:


y_log.skew()


# In[25]:


sns.distplot(y_log)
plt.title(f'Log price with sskew {y_log.skew()}')
plt.show()


# ### Model with log prices
# 
# Logarithms are often used in data output and modeling, especially when the data has a skewed distribution or large outliers. Taking the data logarithm helps to remove the skewness of the distribution, reduce the impact of outliers, and transform the data to better match the assumptions of the statistical model.

# In[26]:


prices = np.log(data['price'])
features = data.drop('price', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])
print(coefficients)


# R2 score is good but we will try to improve this

# # Pvalues & Evaluating Coefficients

# In[27]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}).sort_values(by='pvalues') #OVER 0.05 NOT SIGNIFICANT


# significiant if pvalue < 0.05

# # Testing for multicollinearity

# In[28]:


vif = [variance_inflation_factor(exog=X_incl_const, exog_idx = each) for each in range (X_incl_const.shape[1])]

pd.DataFrame({'coef': X_incl_const.columns, 'vif' : np.round(vif, 2)}).sort_values(by='vif',ascending=False)


# significiant if VIF < 10

# In[29]:


pred2=['curbweight','model','horsepower','doornumber','carbody','stroke','peakrpm','fuelsystem'] 
# LOW pvalue and VIF


# # Model Simplification & The Baysian Information Criterion

# In[30]:


#Original model with log prices and all features

X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

org_coef = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT

print('BIC is:',results.bic)
print('R-squared is:',results.rsquared)


# In[31]:


#Reduced model #1 excluding INDUS

X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['fueltype','compressionratio','horsepower','carlength','wheelbase','highwaympg'], axis=1) #modyfikacja modelu

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

coef_minus_pred = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT

print('BIC is:',results.bic) #lepsze
print('R-squared is:',results.rsquared) #brak wpływu


# In[32]:


frames = [org_coef, coef_minus_pred]
pd.concat(frames, axis=1)


# # Residuals & Residuals Plots

# In[33]:


# Modified model: transformed (using log prices) & simplified (droppint two features)
prices = np.log(data['price']) #LOG PRICES
features = data[pred2]

X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)
 
# Using Statsmodel
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

# Graph of Actual vs Predicter Prices
corr = round(Y_train.corr(results.fittedvalues), 2)
plt.scatter(x=Y_train,y=results.fittedvalues, c = 'indigo', alpha = 0.6)
plt.plot(Y_train,Y_train, color='red')
plt.xlabel('Actual log prices $y_i$', fontsize= 14)
plt.ylabel('Predicted log prices $\hat y_i$', fontsize= 14)
plt.title(f'Actual vs Predicted log prices: $y_i$ vs $ \hat y_1$ (Corr: {corr})',fontsize = 17)
plt.show()

plt.scatter(x=np.e**Y_train,y=np.e**results.fittedvalues, c = 'indigo', alpha = 0.6)
plt.plot(np.e**Y_train,np.e**Y_train, color='red')

plt.xlabel('Actual prices 000s $y_i$', fontsize= 14)
plt.ylabel('Predicted prices 000s $\hat y_i$', fontsize= 14)
plt.title(f'Actual vs Predicted prices: $y_i$ vs $ \hat y_1$ (Corr: {corr})',fontsize = 17)
plt.show()

# Residuals vs Predicted values
plt.scatter(x=results.fittedvalues,y=results.resid, c = 'indigo', alpha = 0.6)

plt.xlabel('Predicted log prices', fontsize= 14)
plt.ylabel('Residuals', fontsize= 14)
plt.title(f'Residuals vs Predicted prices',fontsize = 17)
plt.show()

# Mean Squared Error & R-Squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)


# In[34]:


# Distribution of Residuals (log prices) - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color = 'navy')
plt.title(f'Log price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')


# In[35]:


#ORIGINAL DATA
prices = data['price'] #ORGINAL PRICES
features = data[pred2]

X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)

# Using Statsmodel
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_incl_const)
results = model.fit()
corr = round(Y_train.corr(results.fittedvalues), 2)

# Predicted vs Actual prices
plt.scatter(x=Y_train,y=results.fittedvalues, c = 'teal', alpha = 0.6)
plt.plot(Y_train,Y_train, color='cyan')
plt.xlabel('Actual prices $y_i$', fontsize= 14)
plt.ylabel('Predicted prices $\hat y_i$', fontsize= 14)
plt.title(f'Actual vs Predicted prices: $y_i$ vs $ \hat y_1$ (Corr: {corr})',fontsize = 17)
plt.show()

# Residuals vs Prices
plt.scatter(x=results.fittedvalues,y=results.resid, c = 'teal', alpha = 0.6)

plt.xlabel('Predicted prices', fontsize= 14)
plt.ylabel('Residuals', fontsize= 14)
plt.title(f'Residuals vs Predicted prices',fontsize = 17)
plt.show()

# Distribution of Residuals (log prices) - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color = 'teal')
plt.title(f'Log price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')


# Mean Squared Error & R-Squared
reduced_mse = round(results.mse_resid, 3)
reduced_rsquared = round(results.rsquared, 3)


# # MSE RMSE

# In[36]:


pd.DataFrame({'R-Squared':[reduced_log_rsquared,reduced_rsquared],
              'MSE':[reduced_log_mse,reduced_mse,],
             'RMSE': np.sqrt([reduced_log_mse,reduced_mse])},
             index=['Reduced Log Model','Full Normal Price Model'])


# # Summary

# based on the available R-Squared, MSE and RMSE statistics and the charts presented above, I chose the model with the logarithm of the price. It fits better and the Residual plot looks much better. Residuals are nicely distributed throughout the graph, which proves the lack of autocorrelation, they oscillate around zero. In addition, it is more like a normal distribution because it is less slanted. Higher correlation in the graphs also indicates a better fit of the model.
# 

# In[37]:


upper_bound = np.log(30) + 2*np.sqrt(reduced_log_mse)
lower_bound = np.log(30) - 2*np.sqrt(reduced_log_mse)
print('The upper bound in normal prices is $', np.e**upper_bound * 1000)
print('The lower bound in normal prices is $', np.e**lower_bound * 1000)


# # BUILDING A TOOL
# Since we already have a prepared model, it would be appropriate to use it. We will use a function that will accept arguments relevant to the model and estimate the price of this car. To keep the code readable, I made the tool in a separate python file.

# In[45]:


import car_tool as tl
tl.calc_price(2000,1000,0,4,4.3,10000,1,True)

