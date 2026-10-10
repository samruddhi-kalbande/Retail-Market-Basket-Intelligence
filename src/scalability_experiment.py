from pathlib import Path
import time

import pandas as pd
from scipy.sparse import load_npz
from mlxtend.frequent_patterns import apriori, fpgrowth


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATRIX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "basket_matrix.npz"
)

PRODUCT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_categories.txt"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SAMPLE_SIZES = [
    50_000,
    100_000,
    250_000,
    500_000
]

MIN_SUPPORT = 0.01

MAX_LEN = 3


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("APRIORI VS FP-GROWTH - SCALABILITY EXPERIMENT")
print("=" * 70)

print("\nLoading sparse basket matrix...")

matrix = load_npz(MATRIX_FILE)

print(
    f"Full matrix: "
    f"{matrix.shape[0]:,} baskets × "
    f"{matrix.shape[1]} products"
)


# ---------------------------------------------------------
# LOAD PRODUCT NAMES
# ---------------------------------------------------------

with open(
    PRODUCT_FILE,
    "r",
    encoding="utf-8"
) as file:

    product_names = [
        line.strip()
        for line in file
    ]


# ---------------------------------------------------------
# RUN EXPERIMENTS
# ---------------------------------------------------------

results = []

for sample_size in SAMPLE_SIZES:

    if sample_size > matrix.shape[0]:
        continue

    print("\n" + "=" * 70)
    print(
        f"TESTING {sample_size:,} BASKETS"
    )
    print("=" * 70)

    sample_matrix = matrix[:sample_size]

    basket_data = (
        pd.DataFrame.sparse.from_spmatrix(
            sample_matrix,
            columns=product_names
        )
        .astype(bool)
    )

    # -----------------------------------------------------
    # APRIORI
    # -----------------------------------------------------

    print("\nRunning Apriori...")

    start_time = time.perf_counter()

    apriori_itemsets = apriori(
        basket_data,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN,
        low_memory=True
    )

    apriori_runtime = (
        time.perf_counter() - start_time
    )

    print(
        f"Apriori: "
        f"{len(apriori_itemsets):,} itemsets, "
        f"{apriori_runtime:.4f} seconds"
    )

    # -----------------------------------------------------
    # FP-GROWTH
    # -----------------------------------------------------

    print("\nRunning FP-Growth...")

    start_time = time.perf_counter()

    fp_itemsets = fpgrowth(
        basket_data,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    )

    fp_runtime = (
        time.perf_counter() - start_time
    )

    print(
        f"FP-Growth: "
        f"{len(fp_itemsets):,} itemsets, "
        f"{fp_runtime:.4f} seconds"
    )

    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    results.append(
        {
            "baskets": sample_size,
            "apriori_itemsets": len(
                apriori_itemsets
            ),
            "apriori_runtime_seconds": round(
                apriori_runtime,
                4
            ),
            "fpgrowth_itemsets": len(
                fp_itemsets
            ),
            "fpgrowth_runtime_seconds": round(
                fp_runtime,
                4
            )
        }
    )


# ---------------------------------------------------------
# CREATE RESULTS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# CALCULATE SPEED RATIO
# ---------------------------------------------------------

results_df["apriori_to_fpgrowth_runtime_ratio"] = (
    results_df["fpgrowth_runtime_seconds"]
    / results_df["apriori_runtime_seconds"]
).round(2)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "scalability_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SCALABILITY RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved results to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("SCALABILITY EXPERIMENT COMPLETE")
print("=" * 70)