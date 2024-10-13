import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.dates as mdates

# Шаг 1: Загрузите конфигурацию из файла config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Шаг 2: Используйте путь к файлу из конфигурации
csv_file_path = config['csv_file_path']

# Шаг 3: Загрузите данные из CSV файла
df = pd.read_csv(csv_file_path)

# Преобразуем колонку Date в формат datetime
df['Date'] = pd.to_datetime(df['Date'])

# Шаг 4: Параметр n для скользящего среднего
n = 10  # Здесь можно задать любое значение

# Рассчитайте скользящее среднее по колонке "Close" за n дней
df[f'{n}-day SMA'] = df['Close'].rolling(window=n).mean()

# Округлите значения в колонке n-day SMA до 3 знаков после запятой
df[f'{n}-day SMA'] = df[f'{n}-day SMA'].round(3)

# Шаг 5: Сохраните обновленный DataFrame обратно в CSV файл
df.to_csv(csv_file_path, index=False)

# Построим график (если нужно визуализировать данные)
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df[f'{n}-day SMA'], label=f'{n}-day SMA', linestyle='--')

# Настроим оси и добавим легенду
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Close Price and {n}-day SMA')

# Форматирование меток по оси X для улучшения читаемости
plt.xticks(rotation=45)  # Поворачиваем метки на 45 градусов
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Автоматический выбор частоты меток
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Формат меток даты

plt.tight_layout()

# Отображаем график
plt.show()
