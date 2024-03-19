import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skforecast.model_selection_statsmodels import grid_search_sarimax
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

def add_datetime_features(df):
    """
    Add the columns from dataframe containing different date aspects separately

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    df: Pandas dataframe 
    """
    df["minute"] = df['ds'].dt.minute
    df["hour"] = df['ds'].dt.hour
    df["day"] = df['ds'].dt.day
    df["week"] = df['ds'].dt.isocalendar().week
    df["month"] = df['ds'].dt.month
    df["year"] = df['ds'].dt.year
    return df


urlIter = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/solar_iter_consignas.csv'
urlArico = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/solar_arico_consignas.csv'
urlFuturoIter = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/Febrero%20Todo%20ITER.csv'

df = pd.read_csv(urlIter) # dataframe inicial
df = df[:-96]

dfFutureIter = pd.read_csv(urlFuturoIter)

### Normalizado y limpieza de datos
df['y'] = df['y'].mul(-1)

df = df.drop('Unnamed: 0', axis=1)
dfFutureIter = dfFutureIter.drop('Unnamed: 0', axis=1)

df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
dfFutureIter['ds'] = pd.to_datetime(dfFutureIter['ds'], format='%Y-%m-%d %H:%M:%S')


### Dataframe de testeo que contiene los valores predichos de los regresores
test = dfFutureIter.copy(deep=True)
test = add_datetime_features(test)
test.drop(columns = 'solar_irradiance',inplace=True)
test.rename(columns={'f_solar_irradiance': 'solar_irradiance'}, inplace=True)

first_date = test.iloc[0]['ds']
last_date = test.iloc[-1]['ds']


### Dataframe de entrenamiento con los valores históricos hasta la primera fecha del dataframe de testeo
train_mask = (df['ds'] < first_date)
train = df.loc[train_mask].copy(deep=True)
train = add_datetime_features(train)


### Dataframe que contiene los resultados tanto de la predicción como de los valores reales de la energía
test_mask = (df['ds'] >= first_date) & (df['ds'] <= last_date)
results = df.loc[test_mask].copy(deep=True)

# Serie que contiene las fechas y horas en los que la altitud es menor a 0
night_hours = results.loc[(results['altitude'] < 0)]
night_hours = night_hours.ds

# Serie que contiene las fechas y horas en los que hay consigna
setpoints_hours = results.loc[(results['setpoints'] < 500)]
setpoints_hours = setpoints_hours.ds

# Finalmente nos quedamos solo con las fechas y el valor rel de la energía para comparar con las predicciones
results = results[['ds', 'y']]
results.set_index('ds', inplace=True)


### Declaramos el objeto con los posibles valores de cada parámetro
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

### Establecemos las columnas relevantes para el modelo
relevant_columns = ['solar_irradiance', 'month', 'hour', 'setpoints']

### Dividimos los dataframes de entrenamiento y test en sus variables regresoras y objetivo
trainX, trainY = train.loc[:, relevant_columns], train['y']
testX, testY = test.loc[:, relevant_columns], test['y']

### Inicializamos el modelo y la optimización de parámetros
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Ajustamos todas las combinaciones de parámetros posibles
grid_search.fit(trainX, trainY)

# Obtenemos el mejor modelo
best_grid = grid_search.best_estimator_

# Predecimos los valores de usando los valores de los regresores a futuro (predicciones)
rf_forecast = best_grid.predict(testX)
results['rf_forecast'] = rf_forecast
results['rf_forecast'].loc[results.index.isin(night_hours)] = 0