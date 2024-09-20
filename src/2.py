import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

sleep_data = pd.read_csv('./simplified_ds/simplified_sleep_data.csv')
heart_rate_data = pd.read_csv('./simplified_ds/simplified_heart_rate_data.csv')
calories_data = pd.read_csv('./simplified_ds/simplified_calories.csv')

datasets = [sleep_data, heart_rate_data, calories_data]
dataset_names = ['Sleep', 'Heart Rate', 'Calories']

def ensure_day_no_column(data):
    if 'Day No' not in data.columns:
        print(f"'Day No' column not found in dataset, creating one using row index.")
        data['Day No'] = np.arange(len(data))
    return data

def prepare_data(data):
    X = data['Day No'].values.reshape(-1, 1)
    y = data['Number of entries'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_features(X):
    X_df = pd.DataFrame(X, columns=['Day No'])
    
    X_df['Day of Week'] = X_df['Day No'] % 7
    X_df['Week of Year'] = X_df['Day No'] // 7
    
    return X_df

def train_and_predict(X_train, X_test, y_train, y_test, data, future_days=7):
    models = {
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
        'LightGBM': LGBMRegressor(random_state=42, min_child_samples=1, min_data_in_bin=1),
        'GBR': GradientBoostingRegressor(random_state=42)
    }
    
    last_day = data['Day No'].max()
    future_dates = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
    
    X_train_featured = create_features(X_train)
    X_test_featured = create_features(X_test)
    future_dates_featured = create_features(future_dates)
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train_featured, y_train)
            y_pred = model.predict(X_test_featured)
            mse = mean_squared_error(y_test, y_pred)
            future_predictions = model.predict(future_dates_featured)
        except Exception as e:
            print(f"Error training {name} model: {str(e)}")
            print(f"Skipping {name} model")
            continue

        results[name] = {
            'model': model,
            'mse': mse,
            'predictions': future_predictions
        }
    
    return results, future_dates

def plot_results(datasets, dataset_names, all_results, all_future_dates):
    fig, axs = plt.subplots(len(datasets), 1, figsize=(15, 5 * len(datasets)), sharex=True)
    colors = {'XGBoost': 'r', 'CatBoost': 'g', 'LightGBM': 'b', 'GBR': 'y'}
    
    for i, (data, name, results, future_dates) in enumerate(zip(datasets, dataset_names, all_results, all_future_dates)):
        axs[i].plot(data['Day No'], data['Number of entries'], 'k-', label='Actual')
        
        for model_name, result in results.items():
            axs[i].plot(future_dates, result['predictions'], f'{colors[model_name]}--', label=f'{model_name} (MSE: {result["mse"]:.2f})')
        
        axs[i].set_title(f'{name} Data - Actual vs Predicted')
        axs[i].set_xlabel('Day No.')
        axs[i].set_ylabel('Number of entries')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def save_predictions_to_file(dataset_name, results, future_dates):
    with open(f'{dataset_name.lower().replace(" ", "_")}_predictions.txt', 'w') as f:
        f.write(f"Predictions for the next 7 days ({dataset_name}):\n")
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            for day, pred in zip(future_dates.flatten(), result['predictions']):
                f.write(f"  Day {day}: {pred:.2f}\n")
        f.write(f"\nMean Squared Errors:\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result['mse']:.2f}\n")

# Main 
all_results = []
all_future_dates = []
for data, name in zip(datasets, dataset_names):
    print(f"Processing {name} data...")
    
    data = ensure_day_no_column(data)
    
    X_train, X_test, y_train, y_test = prepare_data(data)
    results, future_dates = train_and_predict(X_train, X_test, y_train, y_test, data)
    all_results.append(results)
    all_future_dates.append(future_dates)
    
    save_predictions_to_file(name, results, future_dates)
    
    print(f"Predictions for {name} data have been saved to {name.lower().replace(' ', '_')}_predictions.txt")
    print()

plot_results(datasets, dataset_names, all_results, all_future_dates)
