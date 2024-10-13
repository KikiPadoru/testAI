import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Загрузите конфигурацию из файла config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Используйте путь к файлу из конфигурации
csv_file_path = config['csv_file_path']

# Загрузите данные из CSV файла
data = pd.read_csv(csv_file_path)

# Убедимся, что дата распознается как дата
data['Date'] = pd.to_datetime(data['Date'])

# Установим индекс на дату
data.set_index('Date', inplace=True)

# Проверим данные
print(data.head())

# Объединим данные о цене закрытия и 10-day SMA в один датафрейм
df = data[['Close', '10-day SMA']].dropna()

# Построим график данных
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['10-day SMA'], label='10-day SMA')
plt.legend()
plt.show()

# Обучаем модель ARIMA
# Для начала попробуем простую ARIMA модель только на основании цены закрытия
model = ARIMA(df['Close'], order=(10, 1, 0))  # Параметры ARIMA можно будет оптимизировать позже
model_fit = model.fit()

# Сделаем прогноз на 10 дней
forecast = model_fit.forecast(steps=10)
print("Прогноз на следующие 10 дней:")
print(forecast)

# Построим график прогноза вместе с реальными данными
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual Close Price')
plt.plot(pd.date_range(df.index[-1], periods=11, freq='D')[1:], forecast, label='Forecasted Close Price', color='red')
plt.legend()
plt.show()

# Оценка модели с использованием среднеквадратичной ошибки (MSE)
train_size = int(len(df) * 0.8)
train, test = df['Close'][0:train_size], df['Close'][train_size:]
history = [x for x in train]
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Построим график реальных данных против прогнозируемых
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Close Price')
plt.plot(test.index, predictions, color='red', label='Predicted Close Price')
plt.legend()
plt.show()
