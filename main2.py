import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import json

# Загрузка конфигурации из config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Путь к CSV-файлу
csv_file_path = config['csv_file_path']

# Загрузка данных
data = pd.read_csv(csv_file_path)

# Пропускаем строки с NaN в ARIMA, если таковые имеются
data = data.dropna(subset=['ARIMA'])

# Используем только 'Close' и 'ARIMA' для прогнозирования
close_prices = data[['Close', 'ARIMA']].values

# Масштабируем данные для LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Подготовка данных для LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])  # Прогнозируем только 'Close'
    return np.array(dataX), np.array(dataY)

time_step = 3  # Используем 10 дней для обучения
X, y = create_dataset(scaled_data, time_step)

# Разделение данных на обучающую и тестовую выборки (70% для обучения)
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Преобразуем данные для LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)  # 2 признака: 'Close' и 'ARIMA'
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# Создаем модель LSTM с регуляризацией
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 2)))
model.add(Dropout(0.3))  # Dropout для предотвращения переобучения
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.3))  # Dropout для предотвращения переобучения
model.add(Dense(units=25, kernel_regularizer='l2'))  # L2 регуляризация
model.add(Dense(units=1))

# Компилируем модель
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучаем модель
model.fit(X_train, y_train, batch_size=32, epochs=30)

# Прогнозирование на тестовой выборке
predictions = model.predict(X_test)

# Преобразуем прогнозы и реальные значения обратно в оригинальный масштаб
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros_like(predictions))))
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1)))))

# Проверяем наличие NaN значений
if np.any(np.isnan(predictions)) or np.any(np.isnan(y_test_actual)):
    print("Предсказания или реальные значения содержат NaN. Удаление NaN значений.")
    valid_indices = ~np.isnan(predictions)[:, 0] & ~np.isnan(y_test_actual)[:, 0]
    predictions = predictions[valid_indices]
    y_test_actual = y_test_actual[valid_indices]

# Вычисляем ошибки
mse = mean_squared_error(y_test_actual[:, 0], predictions[:, 0])
r2 = r2_score(y_test_actual[:, 0], predictions[:, 0])

# Визуализация данных
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# Полный график реальных данных и прогнозов
axs[0].plot(scaler.inverse_transform(scaled_data)[:, 0], label='Реальные цены')
axs[0].plot(range(train_size + time_step, len(scaled_data) - 1), predictions[:, 0], label='Прогнозируемые цены')
axs[0].set_title('Реальные и прогнозируемые цены (весь период)')
axs[0].set_xlabel('Дни')
axs[0].set_ylabel('Цена закрытия')
axs[0].legend()

# График тестовой выборки (более детально)
axs[1].plot(y_test_actual[:, 0], label='Реальные цены')
axs[1].plot(predictions[:, 0], label='Прогнозируемые цены')
axs[1].set_title('Прогнозируемые vs реальные цены (тестовая выборка)')
axs[1].set_xlabel('Дни')
axs[1].set_ylabel('Цена закрытия')
axs[1].legend()

plt.tight_layout()
plt.show()

# Вывод ошибок
print(f"Среднеквадратическая ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R^2): {r2:.4f}")
