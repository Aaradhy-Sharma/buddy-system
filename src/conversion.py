import pandas as pd
from datetime import datetime

def convert_activity_data(file_path):
    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

    # Print column names
    print("Columns in the data:")
    print(data.columns)

    # Check if required columns exist
    required_columns = ['ActivityMinute']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None

    # Convert the 'ActivityMinute' column to datetime
    try:
        data['ActivityMinute'] = pd.to_datetime(data['ActivityMinute'])
    except Exception as e:
        print(f"Error converting 'ActivityMinute' column to datetime: {e}")
        return None

    # Extract the date from the 'ActivityMinute' column
    data['Date'] = data['ActivityMinute'].dt.date
    
    # Group by date and count the entries
    daily_counts = data.groupby('Date').size().reset_index(name='Number of entries')
    
    # Sort the dates and assign day numbers
    daily_counts = daily_counts.sort_values('Date')
    daily_counts['Day No.'] = range(1, len(daily_counts) + 1)
    
    # Select and reorder the columns
    result = daily_counts[['Day No.', 'Number of entries']]
    
    return result

# File path (you would replace this with the path to your actual file)
file_path = 'calories.csv'

# Convert the data
result = convert_activity_data(file_path)

if result is not None:
    # Display the result
    print("\nConverted data:")
    print(result)

    # Optionally, save the result to a new CSV file
    result.to_csv('simplified_calories.csv', index=False)
    print("\nResult saved to 'simplified_activity_data.csv'")
else:
    print("\nFailed to convert data. Please check the errors above.")