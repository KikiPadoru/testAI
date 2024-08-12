import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла TSLA_with_predictions.csv
data = pd.read_csv('TSLA_with_predictions.csv')

# Преобразование столбца Date в тип datetime
data['Date'] = pd.to_datetime(data['Date'])

# Разделение данных на обучающую и тестовую выборки
split_index = int(0.8 * len(data))

# Визуализация результатов
plt.figure(figsize=(18, 6))

# График 1: Весь диапазон данных (обучающая и тестовая выборки)
plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, первый график
plt.plot(data['Date'], data['Close'], color='gray', label='Real Data')
plt.plot(data['Date'], data['Predicted_Close_next_day'], color='red', linestyle='--', label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title('Stock Price Prediction using Random Forest (Full Range)')
plt.legend()

# График 2: Только тестовые данные с отдельной линией для предсказанных цен на следующий день
plt.subplot(1, 2, 2)  # 1 строка, 2 колонки, второй график
plt.plot(data['Date'].iloc[split_index:], data['Close'].iloc[split_index:], color='gray', label='Real Data')
plt.plot(data['Date'].iloc[split_index:], data['Predicted_Close_next_day'].iloc[split_index:], color='red', linestyle='--', label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.title('Stock Price Prediction using Random Forest (Test Data)')
plt.legend()

plt.tight_layout()
plt.show()
