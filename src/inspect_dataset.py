from pathlib import Path

# Location of our extracted dataset
data_folder = Path("data/raw")

print("=" * 60)
print("RETAIL MARKET BASKET INTELLIGENCE")
print("DATASET INSPECTION")
print("=" * 60)

# Find all files inside data/raw
files = list(data_folder.rglob("*"))

# Keep only actual files
files = [file for file in files if file.is_file()]

print(f"\nTotal files found: {len(files)}")

print("\nFiles in the dataset:")
print("-" * 60)

for file in files:
    size_mb = file.stat().st_size / (1024 * 1024)

    print(f"{file.name}")
    print(f"  Extension : {file.suffix}")
    print(f"  Size      : {size_mb:.2f} MB")
    print(f"  Location  : {file}")

print("\n" + "=" * 60)
print("Inspection complete!")
print("=" * 60)