from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ARHyMST_dataset_v1.0"
    / "Data"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CHUNK_SIZE = 200_000

REQUIRED_COLUMNS = [
    "PERSON_PUBLIC_KEY",
    "DATE",
    "CHANNEL",
    "PRODUCT_CATEGORY"
]


# ---------------------------------------------------------
# FIND CSV FILES
# ---------------------------------------------------------

csv_files = sorted(DATA_FOLDER.glob("transactions_part_*.csv"))

print("=" * 70)
print("ARHyMST DATASET - BASKET PREPROCESSING")
print("=" * 70)

print(f"\nCSV files found: {len(csv_files)}")


# ---------------------------------------------------------
# TEMPORARY OUTPUT FILE
# ---------------------------------------------------------

TEMP_FILE = OUTPUT_FOLDER / "cleaned_transactions.csv"

# Remove an old output file if it exists
if TEMP_FILE.exists():
    TEMP_FILE.unlink()


# ---------------------------------------------------------
# PROCESS EACH CSV FILE
# ---------------------------------------------------------

total_rows = 0
cleaned_rows = 0
removed_missing = 0

first_write = True

for file_number, file_path in enumerate(csv_files, start=1):

    print(
        f"\n[{file_number}/{len(csv_files)}] "
        f"Processing {file_path.name}"
    )

    for chunk in pd.read_csv(
        file_path,
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        # -------------------------------------------------
        # REMOVE ROWS WITH MISSING REQUIRED VALUES
        # -------------------------------------------------

        before_cleaning = len(chunk)

        chunk = chunk.dropna(subset=REQUIRED_COLUMNS)

        removed_missing += (
            before_cleaning - len(chunk)
        )

        # -------------------------------------------------
        # REMOVE DUPLICATE PRODUCT RECORDS
        # -------------------------------------------------

        chunk = chunk.drop_duplicates(
            subset=REQUIRED_COLUMNS
        )

        cleaned_rows += len(chunk)

        # -------------------------------------------------
        # CREATE BASKET ID
        # -------------------------------------------------

        chunk["BASKET_ID"] = (
            chunk["PERSON_PUBLIC_KEY"]
            + "_"
            + chunk["DATE"]
            + "_"
            + chunk["CHANNEL"]
        )

        # -------------------------------------------------
        # SAVE CLEANED TRANSACTIONS
        # -------------------------------------------------

        columns_to_save = [
            "BASKET_ID",
            "PERSON_PUBLIC_KEY",
            "DATE",
            "CHANNEL",
            "PRODUCT_CATEGORY"
        ]

        chunk[columns_to_save].to_csv(
            TEMP_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False


# ---------------------------------------------------------
# DISPLAY CLEANING RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING RESULTS")
print("=" * 70)

print(f"\nOriginal records       : {total_rows:,}")
print(f"Rows removed for NA    : {removed_missing:,}")
print(f"Cleaned records        : {cleaned_rows:,}")

print(f"\nCleaned transaction file:")
print(TEMP_FILE)


# ---------------------------------------------------------
# CREATE BASKET SUMMARY
# ---------------------------------------------------------

print("\nCreating basket summary...")

cleaned_data = pd.read_csv(
    TEMP_FILE,
    dtype=str
)

basket_summary = (
    cleaned_data
    .groupby("BASKET_ID")["PRODUCT_CATEGORY"]
    .agg(
        lambda products: " | ".join(
            sorted(set(products))
        )
    )
    .reset_index()
)

basket_summary.columns = [
    "BASKET_ID",
    "PRODUCTS"
]


# ---------------------------------------------------------
# SAVE BASKET DATA
# ---------------------------------------------------------

BASKET_FILE = OUTPUT_FOLDER / "baskets.csv"

basket_summary.to_csv(
    BASKET_FILE,
    index=False
)


# ---------------------------------------------------------
# FINAL INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASKET CREATION COMPLETE")
print("=" * 70)

print(f"\nNumber of baskets : {len(basket_summary):,}")

print(f"\nSaved files:")

print(f"1. {TEMP_FILE}")
print(f"2. {BASKET_FILE}")

print("\nFirst 5 baskets:")
print(basket_summary.head())

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)