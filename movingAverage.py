import pandas as pd
import matplotlib.pyplot as plt

# Загрузите данные из CSV файла
df = pd.read_csv('UBER.csv')

# Рассчитайте скользящее среднее по колонке "Close" за пять дней
df['5-day SMA'] = df['Close'].rolling(window=100).mean()

# Построим график
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['5-day SMA'], label='5-day SMA', linestyle='--')

# Настроим оси и добавим легенду
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Close Price and 5-day SMA')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Отображаем график
plt.show()
