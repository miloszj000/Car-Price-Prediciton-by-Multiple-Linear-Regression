#!/usr/bin/env python
# coding: utf-8

# # Libraries import

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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


data['model'] = [x.split()[0] for x in data['CarName']]
data['model'] = data['model'].replace({'maxda': 'Mazda','mazda': 'Mazda', 
                                     'nissan': 'Nissan', 
                                     'porcshce': 'Porsche','porsche':'Porsche', 
                                     'toyouta': 'Toyota', 'toyota':'Toyota',
                            'vokswagen': 'Volkswagen', 'vw': 'Volkswagen', 'volkswagen':'Volkswagen'})


# In[8]:


data= data.drop(['car_ID','CarName'], axis=1)


# In[9]:


data.price.skew()


# In[10]:


# price distribution
plt.figure(figsize=(10,6))
sns.distplot(data['price'], bins = 30, hist=True, kde=True, color='teal')

plt.xlabel('Price of cars [$]')
plt.ylabel('Count of cars')
plt.title('Count of cars depending of price [$]')
plt.show()


# In[11]:


numerical = data.select_dtypes(include=['int64', 'float64'])


# In[12]:


categorical= data.select_dtypes(include='object')


# In[13]:


categorical


# In[14]:


numerical.corr() #PEARSON CORR

mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,9))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={'size':10})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[15]:


data = data.drop('citympg',axis=1)


# # Data visualisation

# In[16]:


import matplotlib.pyplot as plt

model_counts = data['model'].value_counts()

# Wyświetlenie wyników
print(model_counts)
print()

# Wykres kołowy
plt.pie(model_counts, labels=model_counts.index, autopct='%.1f%%')

plt.title('Count of Cars by Model')

# Wyśrodkowanie wykresu kołowego
plt.gca().set_aspect('equal')
plt.tight_layout()

# Wyświetlenie wykresu kołowego
plt.show()


# Most used cars

# In[17]:


carbody_counts = data['carbody'].value_counts()

# Wyświetlenie wyników
print(carbody_counts)
print()

# Wykres kołowy
plt.pie(carbody_counts, labels=carbody_counts.index, autopct='%.1f%%')

plt.title('Count of Cars by Body Type')

# Wyśrodkowanie wykresu kołowego
plt.gca().set_aspect('equal')
plt.tight_layout()

# Wyświetlenie wykresu kołowego
plt.show()


# In[18]:


enginelocation_counts = data['enginelocation'].value_counts()

# Wyświetlenie wyników
print(enginelocation_counts)
print()

# Wykres kołowy
plt.pie(enginelocation_counts, labels=enginelocation_counts.index, autopct='%.1f%%')

plt.title('Engine Location Counts')
plt.show()


# # Dummies

# In[19]:


df2 = pd.get_dummies(categorical, drop_first=True)
data = pd.concat([numerical,df2],axis=1)
data


# # Model
# basic model without any changes

# In[20]:


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


# ## Pvalues

# In[21]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

#results.params
#results.pvalues
stats = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)})
stats


# Selecting significant variables pvalue < 0.05

# In[22]:


significant_features = stats.drop(stats[stats['pvalues'] > 0.05].index)
significant_features = significant_features.iloc[:, 0].index
print(significant_features) #Filtration ony significant variables


# # Regression using LOG
# Creating model with use log

# In[23]:


prices_log = np.log(data['price']) # LOG PRICES
features_log = data.drop(['price'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features_log, prices_log, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])
print(coefficients)


# # Pvalues

# In[24]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

#results.params
#results.pvalues
pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)})


# In[25]:


data[significant_features].shape


# # Model simplification & Baysian Information Criterion

# In[26]:


prices = data['price']
features = data[significant_features]

X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = round(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']), 3)
print(coefficients)


# The model was created based on only statistically significant predictors. The match on the test set slightly increased.

# # Testing for multicollinearity

# In[27]:


features = data[significant_features]
X_incl_const = sm.add_constant(features)
vif = [variance_inflation_factor(exog=X_incl_const.values, exog_idx=each) for each in range(X_incl_const.shape[1])]
vif_stat = pd.DataFrame({'coef': X_incl_const.columns, 'vif': np.round(vif, 2)})
vif_stat


# # New model without VIF > 10

# In[28]:


significant_features = significant_features.drop(['carlength', 'curbweight']) #droping features with multicollinearity


# In[29]:


prices = data['price']
features = data[significant_features]


X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = round(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']), 3)
print(coefficients)


# 
# the fit on the test set deteriorated, but we got rid of the multicollinearity

# # New log model with significant features

# In[30]:


prices = np.log(data['price'])
features = data[significant_features]


X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))
print('Intercept:', regr.intercept_)
print(f"Model: {regr} and RMSE score: {np.sqrt(mean_squared_error(Y_test, y_pred))}, R2 score: {r2_score(Y_test, y_pred)}")

coefficients = round(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']), 3)
print(coefficients)


# # Residuals

# In[31]:


#ORIGINAL DATA
prices = data['price']
features = data[significant_features]

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
plt.title(f'Price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')


# Mean Squared Error & R-Squared
reduced_mse = round(results.mse_resid, 3)
reduced_rsquared = round(results.rsquared, 3)


# In[32]:


# Modified model: transformed (using log prices) & simplified (droppint 3 features)

prices = np.log(data['price']) #LOG PRICES
features = data[significant_features]

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

# Residuals vs Predicted values
plt.scatter(x=results.fittedvalues,y=results.resid, c = 'indigo', alpha = 0.6)

plt.xlabel('Predicted log prices', fontsize= 14)
plt.ylabel('Residuals', fontsize= 14)
plt.title(f'Residuals vs Predicted prices',fontsize = 17)
plt.show()

# Mean Squared Error & R-Squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)

# Distribution of Residuals (log prices) - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color = 'navy')
plt.title(f'Log price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()


# in both cases, the fit is very good, I did not observe patterns in the distribution of residuals.
# 

# ## Random Forest Method

# In[33]:


rmse_test = []
r2_test = []
model_names = []



X= data[significant_features]
y= data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
models=[rf]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_test.append(round(np.sqrt(mean_squared_error(y_test, y_pred)),2))
    r2_test.append(round(r2_score(y_test, y_pred),2))
    print (f'model : {model} and  rmse score is : {round(np.sqrt(mean_squared_error(y_test, y_pred)),2)}, r2 score is {round(r2_score(y_test, y_pred),4)}')

    


# In[34]:


r_squared_values = [reduced_log_rsquared, reduced_rsquared] + r2_test
rmse_values = [np.sqrt(reduced_log_mse), np.sqrt(reduced_mse)] + rmse_test

results = pd.DataFrame({'R-Squared': r_squared_values,
                        'RMSE': rmse_values},
                       index=['Reduced Log Model', 'Full Normal Price Model', 'Random Forest'])
results


# In[35]:


upper_bound = np.log(13) + 2*np.sqrt(reduced_log_mse)
down_bound = np.log(13)- 2*np.sqrt(reduced_log_mse)

print('The upper bound in log prices for a 95% prediciton interval is ', upper_bound)
print('The lower bound in log prices for a 95% prediciton interval is ', down_bound)


print('The upper bound in prices for a 95% prediciton interval is ', np.e**upper_bound*1000)
print('The lower bound in prices for a 95% prediciton interval is ', np.e**down_bound*1000)


# In[36]:


def predict_price_rf(model, input_data):
    """
    Przewiduje cenę na podstawie modelu lasów losowych i danych wejściowych.

    Parametry:
    - model: Wytrenowany model lasów losowych
    - input_data: DataFrame zawierający dane wejściowe

    Zwraca:
    - Przewidywaną cenę
    """
    input_data = input_data.values.reshape(1, -1)
    predicted_price = model.predict(input_data)
    return predicted_price

input_data = pd.DataFrame({
    'wheelbase': [100],
    'carwidth': [65],
    'enginesize': [200],
    'peakrpm': [6000],
    'fueltype_gas': [1],
    'carbody_hatchback': [1],
    'carbody_sedan': [0],
    'carbody_wagon': [0],
    'enginelocation_rear': [0],
    'model_bmw': [0],
    'model_dodge': [0],
    'model_mercury': [0],
    'model_mitsubishi': [0],
    'model_peugeot': [0],
    'model_plymouth': [0],
    'model_subaru': [1]
})

# Wywołanie funkcji predict_price_rf
predicted_price_rf = predict_price_rf(rf, input_data)

print('Predicted price:', np.round(predicted_price_rf, 3))


# In[ ]:




