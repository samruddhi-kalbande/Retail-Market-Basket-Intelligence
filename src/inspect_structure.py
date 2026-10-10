from pathlib import Path
import pandas as pd

# Location of the extracted transaction files
data_folder = Path("data/raw/ARHyMST_dataset_v1.0/Data")

# Get all CSV files
csv_files = sorted(data_folder.glob("*.csv"))

print("=" * 70)
print("ARHyMST DATASET - STRUCTURE INSPECTION")
print("=" * 70)

print(f"\nNumber of CSV files: {len(csv_files)}")

# Inspect the first CSV file
first_file = csv_files[0]

print(f"\nInspecting file:")
print(first_file)

print("\n" + "-" * 70)
print("FIRST 10 ROWS")
print("-" * 70)

df = pd.read_csv(first_file)

print(df.head(10).to_string())

print("\n" + "-" * 70)
print("COLUMN NAMES")
print("-" * 70)

for column in df.columns:
    print(column)

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print(df.dtypes)

print("\n" + "-" * 70)
print("NUMBER OF ROWS AND COLUMNS")
print("-" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

print(df.isnull().sum())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)