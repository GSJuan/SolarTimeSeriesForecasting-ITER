urlIter = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/solar_iter_consignas.csv'
urlArico = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/solar_arico_consignas.csv'
urlFuturoIter = 'https://raw.githubusercontent.com/alu0101325583/CSVs/main/ITER/Febrero%20Todo%20ITER.csv'

import pandas as pd
import numpy as np

from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import save
from neuralprophet import load

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

def evaluate(predicted, actual):
    
    """
    Prints different metrics related to the quality of a prediction
    
    Parameters:
    -----------
    predictions : pandas.Series
        Pandas series with the model predictions.
    actuals : pandas.Series
        Pandas series with the actual values.
    
    Returns:
    --------
    nothing
    """
    
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]):
        actual = np.array(actual)
        predicted = np.array(predicted)
        
    errors = abs(predicted - actual)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(errors, abs(actual))
        c[c == np.inf] = 0
        #c[c == -np.inf] = 0
        c = np.nan_to_num(c)

    mape = float(100) * np.mean(c)
    accuracy = float(100) - mape
    smape_error = smape(predicted, actual)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('SMAPE: {:0.4f}'.format(smape_error))
    

def smape(predictions, actuals):
    """
    Calculates the SMAPE metric between predictions and actuals given as pandas series.
    
    Parameters:
    -----------
    predictions : pandas.Series
        Pandas series with the model predictions.
    actuals : pandas.Series
        Pandas series with the actual values.
    
    Returns:
    --------
    float
        SMAPE value.
    """
    epsilon = 1e-10  # A small constant to avoid division by zero
    denominator = (np.abs(predictions) + np.abs(actuals)) / 2.0 + epsilon
    numerator = np.abs(predictions - actuals)
    return np.mean(numerator / denominator) * 100


def to_yhat1_along(model, df):
    """
    Takes a df on the format of the neural prophet output (predicted values on different columns for each step), 
    traverses it diagonally and collapses all values into the same column 'yhat1'

    Parameters:
    -----------
    model : Neural Prophet model
        model used to predict, with all the properties of the prediction (lag, forecast horizon,...)
    df: pandas.DataFrame
        Pandas dataframe with the predictions of the model.
    
    Returns:
    --------
    df: pandas.DataFrame
        Pandas dataframe with the predictions of the model, but with all the values in the same column 'yhat1'
    """
    if model.n_lags != 0:
        new_df = pd.DataFrame(index=df)
        x = 0
        for row in range(len(df)):
            if df.loc[row]["yhat1"] == None or np.isnan(np.min(df.loc[row]["yhat1"])):
                x = row + 1
            else:
                break
        for row in range(len(new_df)):
            try:
                new_df.loc[row + x, "yhat1"] = df.loc[row + x]["yhat" + str(row + 1)]
            except:
                pass
        new_df = new_df.dropna(subset=['yhat1'])
        return new_df
    
def grow_dataframe(df, n, frequency, value=0):  
    """
    Grows a pandas DataFrame by adding n rows with new dates and zeros as values to the input DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to be grown.
    n : int
        Number of rows to add to the DataFrame.
    frequency : str or pandas.DateOffset
        Frequency strings can have multiples, such as '5H'. See pandas documentation for more details 
        about valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    value:
        Value to be added to the new rows.
    Returns:
    --------
    pandas.DataFrame
        DataFrame with n new rows added at the end with new dates and zeros as values.
    """

    last_date = df.index[-1]
    
    # Creamos un nuevo rango de fechas a partir de la última fecha
    new_dates = pd.date_range(start=last_date, freq=frequency, periods=n+1)

    # Creamos un diccionario con las columnas y valores a cero
    new_values = {col: value for col in df.columns}

    # Creamos un DataFrame con los nuevos valores y fechas
    df_new = pd.DataFrame(new_values, index=new_dates)
    df_new = df_new[1:]
    
    # Agregamos el nuevo DataFrame al original utilizando el método append
    df_final = pd.concat([df,df_new])
    return df_final
    
    
def future_prediction_validation(model, future_df, actual_df):  
    """
    Predicts values using Neural Prophet model in such a way that uses histrorical values for the lagged features and prediction values for the future regressors

    It is useful if you want to forecast values for a forecast horizon bigger than what the model was trained for. 

    For example, if you trained the model to forecast 1 day ahead, but you want to forecast 7 days ahead this function will use the historical values
    for the previous day and the predicted values of the regressors on the first day to forecast the y value on the first day and so on.
    
    Parameters:
    -----------
    model : Neural Prophet model
        Fitted Neural Prophet model to predict future values.
    future_df : pandas.DataFrame
        DataFrame with future dates and regressors.
    actual_df : pandas.DataFrame
        DataFrame with the historical values on the same dates as future dates.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the predicted values for the dates given in future_df.
    """
    results_df = future_df.copy(deep=True)
    
    batch_size = model.n_forecasts
    total_size = future_df.shape[0]
    
    lags = model.n_lags

    if(lags > 0):
        total_size = total_size - lags
        
    results_df['prediction'] = 0.0
    
    
    batchs = total_size // batch_size
    remaining = total_size % batch_size
    
    #print('Para el total de filas, se ejecutaran ', batchs, ' batchs de tamaño ', batch_size, )
    
    if(batchs >= 1):
        batch = 0
        while(batch <= batchs):
            step = (batch * batch_size)
            
            #print('Prediciendo desde la fila ', step, ' hasta la fila ', step + batch_size + lags, ' teniendo en cuenta que hay', lags, 'lags y que se predicen ', batch_size, ' pasos hacia el futuro')
            
            historic = actual_df[step : step + lags]
            predicted_regressors = future_df[step : step + batch_size]
            window = pd.concat([historic, predicted_regressors])
            
            predicted_values = model.predict(window) 
            predicted_values = to_yhat1_along(model, predicted_values)
            predicted_values = predicted_values['yhat1'].array
            
            results_df.iloc[step : step + batch_size].loc[:, 'prediction'] = predicted_values
            
            batch = batch + 1
                  
        if(remaining > 0):
            
            step = (batch * batch_size)
            rest = actual_df[step:]
            fill = batch_size + lags - rest.shape[0]

            #print('Step: ', step, '  Remaining: ', remaining, '  Tamaño del rest:', rest.shape[0], '  fill:', fill)
            rest.set_index('ds', inplace=True)
            rest = grow_dataframe(rest, fill, '15T')  
            rest.reset_index(inplace=True) 
            rest.rename(columns={'index': 'ds'}, inplace=True)

            predicted_values = model.predict(rest) 
            predicted_values = to_yhat1_along(model, predicted_values)

            predicted_values = predicted_values.iloc[:remaining]

            results_df.iloc[step : step + remaining].loc[:, 'prediction'] = predicted_values['yhat1'].array
    return results_df


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
test.drop(columns = 'solar_irradiance',inplace=True)
test.rename(columns={'f_solar_irradiance': 'solar_irradiance'}, inplace=True)

first_date = test.iloc[0]['ds']
last_date = test.iloc[-1]['ds']


### Dataframe de entrenamiento con los valores históricos hasta la primera fecha del dataframe de testeo
train_mask = (df['ds'] < first_date)
train = df.loc[train_mask].copy(deep=True)


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


### Entrenamiento del modelo Neural Prophet
neural_columns = ['solar_irradiance', 'setpoints', 'y', 'ds']
steps = 4*24
model = NeuralProphet(
    n_forecasts=steps,
    n_lags=steps
)

model = model.add_future_regressor(name='setpoints')
model = model.add_future_regressor(name='solar_irradiance')

df_train, df_test = model.split_df(train.loc[:, neural_columns], freq="15T")
metrics = model.fit(df_train, freq="15T", validation_df=df_test)

#model = load('neural_final_vanilla.np')

lags = model.n_lags

actual_df = pd.concat([train[-lags:], df.loc[test_mask].copy(deep=True)])
data = future_prediction_validation(model, future_df = test.loc[:, neural_columns], actual_df = actual_df.loc[:, neural_columns])

results['neural_vanilla'] = data['prediction'].array
results['neural_vanilla'].loc[results.index.isin(night_hours)] = 0