import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

# Загрузите конфигурацию из файла config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Используйте путь к файлу из конфигурации
csv_file_path = config['csv_file_path']

# Загрузите данные из CSV файла
data = pd.read_csv(csv_file_path)

# Преобразование столбца Date в тип datetime и установка в качестве индекса
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Параметр для предсказания на n дней вперед
n_days = 1  # Измените это значение на любое количество дней вперед

# Используем цену закрытия (Close) как целевую переменную, сдвинутую на n дней вперед
data[f'Close_next_{n_days}_days'] = data['Close'].shift(-n_days)

# Удаляем последние n дней, так как у них нет целевого значения
data = data[:-n_days]

# Добавляем 10-day SMA в список признаков
X = data[['Open', 'High', 'Low', 'Volume']]
y = data[f'Close_next_{n_days}_days']

# Разделение данных на обучающую и тестовую выборки
split_index = int(0.85 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Создание и обучение модели случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Прогнозирование на обучающей и тестовой выборках
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Прогноз на n дней вперед (с использованием последних значений признаков)
last_day_features = X.iloc[-1].to_frame().T  # Преобразуем последний ряд в DataFrame
last_day_prediction = rf.predict(last_day_features)

# Оценка качества модели на тестовой выборке
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Визуализация результатов
plt.figure(figsize=(18, 6))

# График 1: Весь диапазон данных (обучающая и тестовая выборки)
plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, первый график
plt.plot(data.index, data['Close'], color='gray', label='Real Data')

# Для предсказаний на тренировочных данных, индекс совпадает с X_train
train_index = data.index[:split_index]
predicted_train_index = train_index + pd.Timedelta(days=n_days)
plt.plot(predicted_train_index, y_pred_train, color='blue', linestyle='--', label='Predicted Data (Train)')

# Для предсказаний на тестовых данных, смещение на n_days вперед
test_index = data.index[split_index:]
predicted_test_index = test_index + pd.Timedelta(days=n_days)
plt.plot(predicted_test_index, y_pred_test, color='red', linestyle='--', label='Predicted Data (Test)')

# Добавление последнего предсказанного значения на график
plt.plot([data.index[-1] + pd.Timedelta(days=n_days)], last_day_prediction, 'ro', label=f'Prediction (Next {n_days} Days)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title(f'Stock Price Prediction using Random Forest (Full Range, {n_days} Days Ahead)')
plt.legend()

# График 2: Только тестовые данные с отдельной линией для предсказанных цен на n дней вперед
plt.subplot(1, 2, 2)  # 1 строка, 2 колонки, второй график
plt.plot(test_index, y_test, color='gray', label='Real Data')
plt.plot(predicted_test_index, y_pred_test, color='red', linestyle='--', label=f'Predicted Data (Next {n_days} Days)')

# Добавление последнего предсказанного значения на график
plt.plot([data.index[-1] + pd.Timedelta(days=n_days)], last_day_prediction, 'ro', label=f'Prediction (Next {n_days} Days)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title(f'Stock Price Prediction using Random Forest (Test Data, {n_days} Days Ahead)')
plt.legend()

plt.tight_layout()
plt.show()
