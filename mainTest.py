import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import json

# Загрузка конфигурации
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

csv_file_path = config['csv_file_path']

# Загрузка данных
data = pd.read_csv(csv_file_path)
# Пропускаем первые 10 строк данных
start_day = 10
data = data.iloc[start_day:]

# Преобразование столбца Date в тип datetime и установка в качестве индекса
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Параметр для предсказания на n дней вперед
n_days = 1

# Целевая переменная: цена закрытия сдвинутая на n дней вперед
data[f'Close_next_{n_days}_days'] = data['Close'].shift(-n_days)

# Удаляем последние n дней, так как у них нет целевого значения
data = data[:-n_days]

# Выбор признаков
X = data[["Close", "Open", "ARIMA"]].values
y = data[f'Close_next_{n_days}_days'].values

# Нормализация данных
from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Преобразование данных для LSTM
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 1
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

# Разделение на обучающую и тестовую выборки
split_index = int(0.85 * len(X_lstm))
X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]

# Создание модели LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, X_train.shape[2])))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split=0.1)

# Прогнозирование на тестовой выборке
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Преобразование обратно из масштаба
y_pred_test = scaler_y.inverse_transform(y_pred_test)
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_pred_train = scaler_y.inverse_transform(y_pred_train)
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))

# Прогноз на последний день
last_day_features = X_scaled[-time_steps:].reshape(1, time_steps, X_scaled.shape[1])
last_day_prediction = model.predict(last_day_features)
last_day_prediction = scaler_y.inverse_transform(last_day_prediction)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Визуализация результатов
plt.figure(figsize=(18, 6))

# График 1: Весь диапазон данных (обучающая и тестовая выборки)
plt.subplot(1, 2, 1)
plt.plot(data.index[time_steps:split_index+time_steps], y_train, color='gray', label='Real Data (Train)')
plt.plot(data.index[time_steps:split_index+time_steps], y_pred_train, color='blue', linestyle='--', label='Predicted Data (Train)')

plt.plot(data.index[split_index+time_steps:], y_test, color='gray', label='Real Data (Test)')
plt.plot(data.index[split_index+time_steps:], y_pred_test, color='red', linestyle='--', label='Predicted Data (Test)')

# Добавление последнего предсказанного значения на график
plt.plot([data.index[-1] + pd.Timedelta(days=n_days)], last_day_prediction, 'ro', label=f'Prediction (Next {n_days} Days)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title(f'Stock Price Prediction using LSTM (Full Range, {n_days} Days Ahead)')
plt.legend()

# График 2: Только тестовые данные
plt.subplot(1, 2, 2)
plt.plot(data.index[split_index+time_steps:], y_test, color='gray', label='Real Data (Test)')
plt.plot(data.index[split_index+time_steps:], y_pred_test, color='red', linestyle='--', label=f'Predicted Data (Next {n_days} Days)')

# Добавление последнего предсказанного значения на график
plt.plot([data.index[-1] + pd.Timedelta(days=n_days)], last_day_prediction, 'ro', label=f'Prediction (Next {n_days} Days)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title(f'Stock Price Prediction using LSTM (Test Data, {n_days} Days Ahead)')
plt.legend()

plt.tight_layout()
plt.show()
