import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style for better readability
sns.set(style="whitegrid")

# Using encoding='latin-1' to handle special characters if default utf-8 fails
file_path = r"C:\Users\asus\OneDrive\Desktop\miniproject-1\laptop_price - dataset.csv"
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except:
    df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Task 1: Plot the price of all laptops (Distribution)
plt.figure(figsize=(10, 6))
sns.histplot(df['Price (Euro)'], kde=True, color='blue')
plt.title('Distribution of Laptop Prices', fontsize=16)
plt.xlabel('Price (Euro)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('price_distribution.png') # שמירת הגרף
plt.show()


# Task 2: Most expensive company (on average) & Average price per company
# Calculate mean price by company and sort descending
avg_price_by_company = df.groupby('Company')['Price (Euro)'].mean().sort_values(ascending=False)

print("\n--- Average Price per Company ---")
print(avg_price_by_company)

most_expensive_company = avg_price_by_company.idxmax()
print(f"\nThe company with the most expensive laptops on average is: {most_expensive_company}")

# Task 3 + 4: Fix and standardize Operating System (OpSys) names
print("\nUnique OpSys before fix:", df['OpSys'].unique())

def set_os_category(os_name):
    """
    פונקציה לאיחוד שמות של מערכות הפעלה דומות
    """
    if 'Windows' in os_name:
        return 'Windows'
    elif 'Mac' in os_name or 'macOS' in os_name:
        return 'Mac'
    elif 'Linux' in os_name:
        return 'Linux'
    elif 'Android' in os_name:
        return 'Android'
    elif 'Chrome' in os_name:
        return 'Chrome OS'
    else:
        return 'No OS/Other'

df['OpSys'] = df['OpSys'].apply(set_os_category)
print("Unique OpSys after fix:", df['OpSys'].unique())

# Task 5: Plot price distribution for each Operating System type (Fixed Layout)

unique_os = df['OpSys'].unique()

# Change to 1 row, multiple columns, and SHARE the Y-axis
fig, axes = plt.subplots(nrows=1, ncols=len(unique_os), figsize=(24, 5), sharey=True)

for i, os in enumerate(unique_os):
    subset = df[df['OpSys'] == os]
    
    # Plot histogram with KDE
    # We turn off the y-label for all but the first graph to reduce clutter
    sns.histplot(subset['Price (Euro)'], ax=axes[i], kde=True)
    
    axes[i].set_title(f'Distribution: {os}')
    axes[i].set_xlabel('Price (Euro)')
    
    # Only show Y label on the very first graph
    if i == 0:
        axes[i].set_ylabel('Number of Laptops')
    else:
        axes[i].set_ylabel('')

plt.suptitle('Price Distribution by OS Category', fontsize=16)
plt.tight_layout()
plt.savefig('opsys_distribution_row.png')
plt.show()

# Task 6: Relationship between RAM and Price & Outlier Detection
plt.figure(figsize=(10, 6))
sns.boxplot(x='RAM (GB)', y='Price (Euro)', data=df)
plt.title('Relationship between RAM and Price (with Outliers)', fontsize=16)
plt.xlabel('RAM (GB)', fontsize=12)
plt.ylabel('Price (Euro)', fontsize=12)
plt.savefig('ram_vs_price.png')
plt.show()

# Statistical Outlier Detection using the IQR Method
Q1 = df['Price (Euro)'].quantile(0.25)
Q3 = df['Price (Euro)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_count = df[(df['Price (Euro)'] < lower_bound) | (df['Price (Euro)'] > upper_bound)].shape[0]
print(f"\nOutlier Detection (IQR Method):")
print(f"Prices above {upper_bound:.2f} Euro are considered outliers.")
print(f"Number of outliers detected: {outliers_count}")

# Task 7: Create a new column 'Storage Type' based on 'Memory' column
def extract_storage_type(memory_str):
    """
    Extracts the storage type (SSD, HDD, Flash Storage, Hybrid) from the Memory column string.
    If multiple types exist (e.g., '128GB SSD + 1TB HDD'), it returns both joined by ' + '.
    """
    storage_types = []
    if 'SSD' in memory_str:
        storage_types.append('SSD')
    if 'HDD' in memory_str:
        storage_types.append('HDD')
    if 'Flash Storage' in memory_str:
        storage_types.append('Flash Storage')
    if 'Hybrid' in memory_str:
        storage_types.append('Hybrid')
    
    if not storage_types:
        return 'Unknown'
    
    return ' + '.join(storage_types)

df['Storage Type'] = df['Memory'].apply(extract_storage_type)

print("\nFirst 5 rows with new 'Storage Type' column:")
print(df[['Memory', 'Storage Type']].head())