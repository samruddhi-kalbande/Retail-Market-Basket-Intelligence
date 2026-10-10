from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, save_npz


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baskets.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CHUNK_SIZE = 100_000


# ---------------------------------------------------------
# PRODUCT CATEGORIES
# ---------------------------------------------------------

PRODUCT_CATEGORIES = [
    "BAKERY PRODUCTS",
    "BEAUTY AND PERSONAL CARE",
    "BEER AND ALCOPOPS",
    "BREAKFAST PRODUCTS",
    "CANNED FOODS",
    "CHEESE",
    "CHILDREN'S WORLD (KIDS' PRODUCTS)",
    "CLOTHING / APPAREL TEXTILES",
    "CONSUMER ELECTRONICS / TECHNOLOGY",
    "DAIRY PRODUCTS",
    "DECOR AND PERFUMES",
    "DECORATIVE AND GIFT PROGRAM",
    "DELICATESSEN - PROCESSED MEAT PRODUCTS",
    "DIY, AUTOMOTIVE AND FURNITURE",
    "DRIED FRUITS AND VEGETABLES",
    "ECONOMAT / STORE SUPPLIES",
    "FLOWERS",
    "FRESH FISH",
    "FRESH MEAT (COUNTER / UNPACKAGED)",
    "FRESH PACKAGED MEAT",
    "FROZEN PRODUCTS",
    "FRUITS",
    "GASTRO PROGRAM (FOOD SERVICE PRODUCTS)",
    "HEALTH FOOD",
    "HOME APPLIANCES",
    "HOME TEXTILES",
    "HOUSEHOLD EQUIPMENT",
    "HOUSEHOLD ESSENTIALS AND CANDLES",
    "HYGIENE AND PAPER PRODUCTS",
    "INGREDIENTS FOR MEAL PREPARATION",
    "KIOSK PRODUCTS",
    "LAUNDRY AND CLEANING PRODUCTS",
    "NON-ALCOHOLIC BEVERAGES",
    "PACKAGING",
    "PARTY SUPPLIES",
    "PET SUPPLIES",
    "POULTRY",
    "SCHOOL, OFFICE AND MEDIA SUPPLIES",
    "SPORTS AND LEISURE",
    "SWEETS AND SNACKS",
    "TABLEWARE / KITCHENWARE",
    "TOYS",
    "VEGETABLES",
    "WHOLESALE TRANSIT",
    "WHOLESALE VP TRANSIT",
    "WINE AND SPIRITS",
]


# ---------------------------------------------------------
# PRODUCT TO COLUMN MAPPING
# ---------------------------------------------------------

product_to_column = {
    product: index
    for index, product in enumerate(PRODUCT_CATEGORIES)
}


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

print("=" * 70)
print("FULL ONE-HOT ENCODING")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_FILE)

print(f"\nExpected product categories: {len(PRODUCT_CATEGORIES)}")
print(f"Chunk size: {CHUNK_SIZE:,}")


# ---------------------------------------------------------
# PROCESS BASKETS IN CHUNKS
# ---------------------------------------------------------

sparse_chunks = []

total_baskets = 0

for chunk_number, baskets in enumerate(
    pd.read_csv(
        INPUT_FILE,
        dtype=str,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"\nProcessing chunk {chunk_number} "
        f"({len(baskets):,} baskets)"
    )

    rows = []
    columns = []

    for row_number, product_string in enumerate(
        baskets["PRODUCTS"]
    ):

        products = product_string.split(" | ")

        for product in products:

            if product in product_to_column:

                rows.append(row_number)

                columns.append(
                    product_to_column[product]
                )

    data = [1] * len(rows)

    sparse_chunk = csr_matrix(
        (
            data,
            (rows, columns)
        ),
        shape=(
            len(baskets),
            len(PRODUCT_CATEGORIES)
        ),
        dtype="int8"
    )

    sparse_chunks.append(sparse_chunk)

    total_baskets += len(baskets)


# ---------------------------------------------------------
# COMBINE ALL CHUNKS
# ---------------------------------------------------------

print("\nCombining sparse matrices...")

full_matrix = csr_matrix(
    pd.concat(
        [
            pd.DataFrame(chunk.toarray())
            for chunk in []
        ]
    )
) if False else None

# Use scipy's vertical stacking directly
from scipy.sparse import vstack

full_matrix = vstack(sparse_chunks, format="csr")


# ---------------------------------------------------------
# SAVE SPARSE MATRIX
# ---------------------------------------------------------

OUTPUT_MATRIX = (
    OUTPUT_FOLDER
    / "basket_matrix.npz"
)

save_npz(
    OUTPUT_MATRIX,
    full_matrix
)


# ---------------------------------------------------------
# SAVE PRODUCT NAMES
# ---------------------------------------------------------

OUTPUT_PRODUCTS = (
    OUTPUT_FOLDER
    / "product_categories.txt"
)

with open(
    OUTPUT_PRODUCTS,
    "w",
    encoding="utf-8"
) as file:

    for product in PRODUCT_CATEGORIES:
        file.write(product + "\n")


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ONE-HOT ENCODING COMPLETE")
print("=" * 70)

print(f"\nNumber of baskets : {full_matrix.shape[0]:,}")
print(f"Number of products: {full_matrix.shape[1]:,}")

print(
    f"Number of non-zero values: "
    f"{full_matrix.nnz:,}"
)

print(
    f"Sparsity: "
    f"{100 * (1 - full_matrix.nnz / (
        full_matrix.shape[0] * full_matrix.shape[1]
    )):.2f}%"
)

print("\nSaved files:")

print(f"1. {OUTPUT_MATRIX}")
print(f"2. {OUTPUT_PRODUCTS}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)