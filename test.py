import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
data = pd.read_csv('TSLA.csv')

# Преобразование столбца Date в тип datetime (если необходимо)
data['Date'] = pd.to_datetime(data['Date'])

# Создание новой колонки с целевой переменной (цена закрытия на следующий день)
data['Close_next_day'] = data['Close'].shift(-1)

# Удаляем последнюю строку, так как для неё нет целевой переменной
data = data[:-1]

# Используем текущие значения признаков для предсказания цены закрытия на следующий день
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close_next_day']

# Разделение данных на обучающую и тестовую выборки
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Создание и обучение модели случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Прогнозирование на всей выборке
data['Predicted_Close_next_day'] = rf.predict(X)

# Сохранение данных с новой колонкой в CSV-файл
data.to_csv('TSLA_with_predictions.csv', index=False)
