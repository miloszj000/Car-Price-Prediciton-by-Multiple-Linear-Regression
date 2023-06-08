import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


data_tool = pd.read_csv('CarPrice_Assignment.csv')

lab=LabelEncoder()
data_tool['fuelsystem']=lab.fit_transform(data_tool['fuelsystem'])
data_tool['cylindernumber']=lab.fit_transform(data_tool['cylindernumber'])
data_tool['enginetype']=lab.fit_transform(data_tool['enginetype'])
data_tool['enginelocation']=lab.fit_transform(data_tool['enginelocation'])
data_tool['drivewheel']=lab.fit_transform(data_tool['drivewheel'])
data_tool['carbody']=lab.fit_transform(data_tool['carbody'])
data_tool['doornumber']=lab.fit_transform(data_tool['doornumber'])
data_tool['aspiration']=lab.fit_transform(data_tool['aspiration'])
data_tool['fueltype']=lab.fit_transform(data_tool['fueltype'])

features_tool = pd.DataFrame({
    'curbweight': data_tool['curbweight'],
    'horsepower': data_tool['horsepower'],
    'doornumber': data_tool['doornumber'],
    'carbody': data_tool['carbody'],
    'stroke': data_tool['stroke'],
    'peakrpm': data_tool['peakrpm'],
    'fuelsystem': data_tool['fuelsystem']})
log_prices_tool = np.log(data_tool['price'])
target_tool = pd.DataFrame(log_prices_tool, columns = ['price'])


curbweight_IDX = 0
horsepower_IDX = 1
doornumber_IDX = 2 
carbody_IDX = 3 
stroke_IDX = 4
peakrpm_IDX = 5
fuelsystem_IDX = 6

car_stats = features_tool.mean().values.reshape(1, 7)



regr = LinearRegression().fit(features_tool, target_tool)
fitted_vals = regr.predict(features_tool)

MSE = mean_squared_error(target_tool, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(curbweight,
                    horsepower,
                    doornumber,
                    carbody,
                    stroke,
                    peakrpm,
                    fuelsystem,
                    high_confidence=True):
    
    # Configure property
    car_stats[0][curbweight_IDX] = curbweight
    car_stats[0][horsepower_IDX] = horsepower
    car_stats[0][stroke_IDX] = stroke
    car_stats[0][peakrpm_IDX] = peakrpm
    car_stats[0][fuelsystem_IDX] = fuelsystem
    car_stats[0][doornumber_IDX] = doornumber
    car_stats[0][carbody_IDX] = carbody
    
    # Make prediction
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



def calc_price(curb_weight, 
                horse_power, 
                door_number, 
                car_body,
                str_oke,
                peak_rpm,
                fuel_system,
                large=True):
    """Estimate the price of a car.
    
    Keyword arguments:
    
    curb_weight -- weight of a vehicle
    
    horse_power -- HP
    
    door_number -- 0 for 2 doors, 1 for 4 doors
    
    car_body    -- 0 : convertible | 1 : hatchback  | 2 : sedan | 3 : wagon | 4: hardtop
    
    str_oke     -- [2.68  3.47  3.4   2.8   3.19  3.39  3.03  3.11  3.23  3.46  3.9   3.41
                    3.07  3.58  4.17  2.76  3.15  3.255 3.16  3.64  3.1   3.35  3.12  3.86
                    3.29  3.27  3.52  2.19  3.21  2.9   2.07  2.36  2.64  3.08  3.5   3.54 
                    2.87 ]
                    
    peak_rpm    -- maximum rotational speed
    
    fuel_system -- 0 : GAS | 1 : DIESEL
    
    large_range -- True for 95% prediction interval, False for a 68% prediction interval
    
    """
    
    

    
    log_est, upper, lower, conf,= get_log_estimate(curbweight = curb_weight, 
                        horsepower = horse_power, 
                        doornumber = door_number, 
                        carbody= car_body,
                       stroke = str_oke,
                       peakrpm = peak_rpm,
                       fuelsystem = fuel_system,
                       high_confidence=large)
    #SCALE
    dollar_est = np.e**log_est 
    dollar_hi = np.e**upper
    dollar_low = np.e**lower
    
    #ROUND VALUES
    dollar_est = np.around(dollar_est, -3)
    dollar_hi = np.around(dollar_hi, -3)
    dollar_low = np.around(dollar_low, -3)
    
    print(f'Estimated price of car: {dollar_est}$\n Upper Bound: {dollar_hi}$\n Lower Bound: {dollar_low}$')
    
    
 





























