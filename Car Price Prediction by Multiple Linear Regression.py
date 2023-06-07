#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# In this project, we will deal with the prediction of car prices, based on their characteristics, using multiple regression.

# # LIBRARIES IMPORT

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython import display


get_ipython().run_line_magic('matplotlib', 'inline')


# # DATA IMPORT

# In[2]:


data = pd.read_csv('CarPrice_Assignment.csv', usecols = ['price', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight', 'enginesize', 'boreratio', 'stroke', 'horsepower','peakrpm', 'citympg', 'highwaympg'])


# # DATA MEANING

# In[3]:


display.Image("dict.jpg")


# In[4]:


data.info()


# # DATA EXPLORATION

# In[5]:


data.head(5)


# In[6]:


data.count()


# # CHECK FOR MISSING VALUES

# In[7]:


pd.isnull(data).any()


# In[8]:


pd.isna(data).any()


# ### We didn't find any missing values

# # Visualising data

# In[9]:


# price distribution
plt.figure(figsize=(10,6))
sns.distplot(data['price'], bins = 30, hist=True, kde=True, color='teal')

plt.xlabel('Price of cars [$]')
plt.ylabel('Count of cars')
plt.title('Count of cars depending of price [$]')
plt.show()


# From the graph it was observed that the vast majority of cars are in the range of 5 - 20 thousand dollars.
# In addition, it was observed that prices start just from 5 thousand. In our dataset, gas-powered cars predominate.

# # CORRELATION
#     correlation is really the most important for us, because it is according to it that we determine which of the predictors will have essential contribution in the construction of our model, in addition, thanks to the correlation matrix we can deduce which predictors may be problems with, for example, by correlation with other predictors. It is worth adding that we take into account only constant variables.

# In[10]:


data.corr() #PEARSON CORR

mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,9))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={'size':10})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# The most correlated with the price are data on car dimensions, kerb weight, engine size, cylinder diameter ratio, horsepower. On the other hand, the inverse correlation of Price occurs with combustion in the city and on the highway. Our predictors are strongly correlated, ( Especially the combustion variables are strongly inversely correlated with the rest of the variables) but at this stage we will not worry about it, but we will keep it in mind.

# # Training & test dataset Split

# In[11]:


prices = data['price']
features = data.drop('price', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)


# # MUTIVARIABLE REGRESSION

# In[12]:


X_incl_const = sm.add_constant(X_train)

regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns = ['coef'])


# The match result on the test data is not satisfactory enough, so we will perform transformations.

# # Transformations

# In[13]:


data['price'].skew() # data is 'tilted' to the right


# In[14]:


price_log = np.log(data['price'])
price_log.skew()


# In[15]:


sns.distplot(price_log)
plt.title(f'Log price with sskew {price_log.skew()}')
plt.show()


# # REGRESSION using LOG

# In[16]:


prices = np.log(data['price']) #LOG PRICES
features = data.drop(['price'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns = ['coef'])


# the data matching on the test set has improved, but we can still work on improving the model.

# In[17]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

#results.params
#results.pvalues
pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT


# as statistically useless we can consider wheelbase, carlenght, carheight, boreratio and peakrmp.

# # TESTING FOR MULTICOLLINEARITY

# In[18]:


vif = [variance_inflation_factor(exog=X_incl_const, exog_idx = each) for each in range (X_incl_const.shape[1])]

pd.DataFrame({'coef': X_incl_const.columns, 'vif' : np.round(vif, 2)})


# In our case, there has been multicollinearity (VIF ~ 10), as a result, it will be necessary to get rid of some predictors.

# # Model Simplification & Baysian Information Criterion

# In[19]:


#Original model with Log prices and all features
X_incl_const = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

org_coef = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT

print('BIC is:',results.bic)
print('R-squared is:',results.rsquared)


# In[26]:


X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['carlength','wheelbase','boreratio','carheight','peakrpm','citympg'], axis=1)

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

coef_minus_indus = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT

print('BIC is:',results.bic) #lepsze
print('R-squared is:',results.rsquared) 


# In[28]:


#Reduced model #1 excluding AGE

X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['highwaympg','citympg','carlength'], axis=1) 

model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

coef_minus_age = pd.DataFrame({'coef': results.params,'pvalues': round(results.pvalues, 3)}) #OVER 0.05 NOT SIGNIFICANT

print('BIC is:',results.bic) #lepsze
print('R-squared is:',results.rsquared)


# After removing the strongly correlated variables and not significant variables, we got better results without losing the R-squared fit.

# # RESIDUALS

# In[32]:


# Modified model: transformed (using log prices) & simplified (droppint 3 features)

prices = np.log(data['price']) #LOG PRICES
features = data.drop(['carlength','wheelbase','boreratio','carheight','peakrpm','citympg'], axis=1)

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


# When I look at the residue graph, I don't notice the distribution pattern, at the densest point they are distributed almost symmetrically around zero.

# In[33]:


# Distribution of Residuals (log prices) - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color = 'navy')
plt.title(f'Log price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')


# In[34]:


#ORIGINAL DATA
prices = data['price']
features = data.drop('price', axis=1)

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


# In[35]:


pd.DataFrame({'R-Squared':[reduced_log_rsquared,reduced_rsquared],
              'MSE':[reduced_log_mse,reduced_mse],
             'RMSE': np.sqrt([reduced_log_mse,reduced_mse])},
             index=['Reduced Log Model','Full Normal Price Model'])


# So now after receiving the right model, we can prepare a tool with which we will be able to determine the approximate price of the car based on predictors.

# # BUILDING A TOOL

# In[45]:


features = data.drop(['carlength','wheelbase','boreratio','carheight','peakrpm','citympg','price'], axis=1)

log_prices = np.log(data['price'])
target = pd.DataFrame(log_prices, columns = ['price'])


# In[47]:


features.head(0)


# In[117]:


carwidth_IDX = 0
curbweight_IDX = 1
enginesize_IDX = 2
stroke_IDX = 3
horsepower_IDX = 4
highwaympg_IDX = 5

car_stats = features.mean().values.reshape(1, 6)


# In[49]:


features.mean().values.shape


# In[64]:


regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


# In[85]:


MSE


# In[120]:


def get_log_estimate(width,
                    curbweight,
                    enginesize,
                    stroke,
                    horsepower,
                    highwaympg,
                    high_confidence=True):
    
    # Configure property
    car_stats[0][carwidth_IDX] = width
    car_stats[0][curbweight_IDX] = curbweight
    car_stats[0][enginesize_IDX] = enginesize
    car_stats[0][stroke_IDX] = stroke
    car_stats[0][horsepower_IDX] = horsepower
    car_stats[0][highwaympg_IDX] = highwaympg
                 
    log_estimate = regr.predict(car_stats)[0][0]
                 
     # Calc Range 
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68 
    
    return log_estimate, upper_bound, lower_bound, interval


# In[131]:


def get_dollar_estimate(width_,curb_weight,engine_size,stroke_,horse_power,highway_mpg,large_range):
    """Estimate the price of a car.
    
    Keyword arguments:
    width_ -- Width of the car
    curb_weight -- Vehicle weight
    engine_size -- Engine displacement
    stroke_ -- Piston stroke
    horse_power -- Number of horsepower
    highway_mpg -- Combustion on the highway
    large_range -- True for 95% prediction interval, False for a 68% prediction interval
    """
    
    
    
    
    log_est, upper, lower, conf,= get_log_estimate(width=width_, 
                                                   curbweight = curb_weight, 
                                                   enginesize = engine_size,
                                                   stroke = stroke_,
                                                   horsepower = horse_power,
                                                   highwaympg = highway_mpg,
                                                   high_confidence = large_range)
    #SCALE FACTOR
    dollar_est = np.e**log_est 
    dollar_hi = np.e**upper
    dollar_low = np.e**lower
    
    #ROUND VALUES
    dollar_est = np.around(dollar_est, -3)
    dollar_hi = np.around(dollar_hi, -3)
    dollar_low = np.around(dollar_low, -3)
    
    print(f'Estimated price of house: {dollar_est}$\n Upper Bound: {dollar_hi}$\n Lower Bound: {dollar_low}$')


# In[137]:


get_dollar_estimate(70,2562,130,2.40,1000,1,True)

