import json
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Игнорируем предупреждения, связанные с моделями ARIMA
warnings.filterwarnings("ignore")

# Шаг 1: Загрузка конфигурации из файла config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Используйте путь к файлу из конфигурации
csv_file_path = config['csv_file_path']

# Шаг 2: Загрузка данных из CSV файла
data = pd.read_csv(csv_file_path, parse_dates=['Date'], index_col='Date')

# Используем колонку "Close" для прогнозирования
closing_prices = data['Close']

# Инициализация столбца для прогнозов
data['ARIMA'] = None

# Шаг 3: Прогнозирование начиная с 10-го дня
start_day = 10
for i in range(start_day, len(closing_prices)):
    train_data = closing_prices[:i]  # данные до текущего дня

    # Обучение модели ARIMA
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()

    # Прогноз на следующий день
    forecast = model_fit.forecast(steps=1).iloc[0]  # Используем .iloc[0] для доступа к значению

    # Запись прогноза в столбец ARIMA на текущий день
    data.iloc[i, data.columns.get_loc('ARIMA')] = forecast

# Шаг 4: Визуализация прогнозов и реальных данных
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Actual')
plt.plot(data.index, data['ARIMA'], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Forecasted Close Prices')
plt.legend()
plt.grid(True)
plt.show()

# Шаг 5: Сохранение данных с прогнозами обратно в исходный CSV файл
data.to_csv(csv_file_path)
