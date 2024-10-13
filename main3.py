import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader

# Load the configuration from the config.json file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Use the file path from the configuration
csv_file_path = config['csv_file_path']

# Load data from the CSV file
data = pd.read_csv(csv_file_path)

# Preprocessing the data
data['Date'] = pd.to_datetime(data['Date'])
data['time_idx'] = (data['Date'] - data['Date'].min()).dt.days
data['group'] = 'stock'  # Assuming it's for one stock, you can adjust if more

# Define the target column
target = 'Close'

# Create TimeSeriesDataset
max_encoder_length = 30  # Historical data for input
max_prediction_length = 1  # Forecast for the next day

training_cutoff = data['time_idx'].max() - max_prediction_length
training_data = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx='time_idx',
    target=target,
    group_ids=['group'],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_reals=['time_idx'],
    time_varying_unknown_reals=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
    target_normalizer=GroupNormalizer(groups=['group']),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation data
validation_data = TimeSeriesDataSet.from_dataset(training_data, data, min_prediction_idx=training_cutoff + 1)

# Create dataloaders
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=batch_size)

# Define the model
tft = TemporalFusionTransformer.from_dataset(
    training_data,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Quantile output
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Training the model
trainer = torch.optim.Adam(tft.parameters(), lr=0.03)
tft.fit(train_dataloader, max_epochs=30, trainer=trainer)

# Forecast on the next day
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)

# Print predicted and actual values
print(f"Predictions: {predictions}")
print(f"Actuals: {actuals}")
