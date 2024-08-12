import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('UBER.csv')

# Преобразование столбца Date в тип datetime и установка в качестве индекса
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Используем цену закрытия (Close) как целевую переменную, сдвинутую на один день вперед
data['Close_next_day'] = data['Close'].shift(-1)

# Удаляем последний день, так как у него нет целевого значения
data = data[:-1]

# Используем текущие значения признаков для предсказания цены на следующий день
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close_next_day']

# Разделение данных на обучающую и тестовую выборки
split_index = int(0.7 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Создание и обучение модели случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Прогнозирование на обучающей и тестовой выборках
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

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
plt.plot(train_index, y_pred_train, color='blue', linestyle='--', label='Predicted Data (Train)')

# Для предсказаний на тестовых данных, индекс совпадает с X_test
test_index = data.index[split_index:]
plt.plot(test_index, y_pred_test, color='red', linestyle='--', label='Predicted Data (Test)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title('Stock Price Prediction using Random Forest (Full Range)')
plt.legend()

# График 2: Только тестовые данные с отдельной линией для предсказанных цен на следующий день
plt.subplot(1, 2, 2)  # 1 строка, 2 колонки, второй график
plt.plot(test_index, y_test, color='gray', label='Real Data')
plt.plot(test_index, y_pred_test, color='red', linestyle='--', label='Predicted Data (Next Day)')

plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title('Stock Price Prediction using Random Forest (Test Data)')
plt.legend()

plt.tight_layout()
plt.show()
