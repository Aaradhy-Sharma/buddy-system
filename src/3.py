import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

sleep_data = pd.read_csv('simplified_sleep_data.csv')
heart_rate_data = pd.read_csv('simplified_heart_rate_data.csv')
calories_data = pd.read_csv('simplified_calories.csv')

datasets = [sleep_data, heart_rate_data, calories_data]
dataset_names = ['Sleep', 'Heart Rate', 'Calories']

def prepare_data(data):
    X = data['Day No.'].values.reshape(-1, 1)
    y = data['Number of entries'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(X_train, X_test, y_train, y_test, future_days=7):
    models = {
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
        'LightGBM': LGBMRegressor(random_state=42, min_child_samples=1, min_data_in_bin=1),
        'GBR': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            future_predictions = model.predict(np.arange(len(X_train) + len(X_test), len(X_train) + len(X_test) + future_days).reshape(-1, 1))
        except Exception as e:
            print(f"Error training {name} model: {str(e)}")
            print(f"Falling back to LinearRegression for {name}")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            future_predictions = model.predict(np.arange(len(X_train) + len(X_test), len(X_train) + len(X_test) + future_days).reshape(-1, 1))

        results[name] = {
            'model': model,
            'mse': mse,
            'predictions': future_predictions
        }
    
    return results

def plot_results(datasets, dataset_names, all_results):
    fig, axs = plt.subplots(len(datasets), 1, figsize=(15, 5 * len(datasets)), sharex=True)
    colors = {'XGBoost': 'r', 'CatBoost': 'g', 'LightGBM': 'b', 'GBR': 'y'}
    
    for i, (data, name, results) in enumerate(zip(datasets, dataset_names, all_results)):
        axs[i].plot(data['Day No.'], data['Number of entries'], 'k-', label='Actual')
        
        for model_name, result in results.items():
            future_days = np.arange(len(data), len(data) + len(result['predictions']))
            axs[i].plot(future_days, result['predictions'], f'{colors[model_name]}--', label=f'{model_name} (MSE: {result["mse"]:.2f})')
        
        axs[i].set_title(f'{name} Data - Actual vs Predicted')
        axs[i].set_xlabel('Day No.')
        axs[i].set_ylabel('Number of entries')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def save_predictions_to_file(dataset_name, results):
    with open(f'{dataset_name.lower().replace(" ", "_")}_predictions.txt', 'w') as f:
        f.write(f"Predictions for the next 7 days ({dataset_name}):\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result['predictions'].tolist()}\n")
        f.write(f"\nMean Squared Errors:\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result['mse']:.2f}\n")

all_results = []
for data, name in zip(datasets, dataset_names):
    print(f"Processing {name} data...")
    X_train, X_test, y_train, y_test = prepare_data(data)
    results = train_and_predict(X_train, X_test, y_train, y_test)
    all_results.append(results)
    
    save_predictions_to_file(name, results)
    
    print(f"Predictions for {name} data have been saved to {name.lower().replace(' ', '_')}_predictions.txt")
    print()

plot_results(datasets, dataset_names, all_results)