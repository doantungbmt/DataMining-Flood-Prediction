import pandas as pd

# Load the data
df = pd.read_csv('water_data_full_combined.csv')

# Display basic info
print("--- Data Info ---")
print(df.info())

# Target variable candidates
print("\n--- Value Counts for potential targets ---")
if 'Cảnh báo value (0-4)' in df.columns:
    print("\nCảnh báo value (0-4):")
    print(df['Cảnh báo value (0-4)'].value_counts(dropna=False))

if 'Mã Cảnh báo' in df.columns:
    print("\nMã Cảnh báo:")
    print(df['Mã Cảnh báo'].value_counts(dropna=False))

# Check for missing values in key columns
key_columns = ['Mực nước (m)', 'BĐ1 (m)', 'BĐ2 (m)', 'BĐ3 (m)', 'Dung tích (m3)', 'Q đến (m3/s)', 'Q xả (m3/s)']
print("\n--- Missing values in key columns ---")
for col in key_columns:
    if col in df.columns:
        print(f"{col}: {df[col].isnull().sum()}")

# Sample of data
print("\n--- Sample data ---")
print(df.head())
