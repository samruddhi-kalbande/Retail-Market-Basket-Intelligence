from pathlib import Path
import time

import pandas as pd
from scipy.sparse import load_npz
from mlxtend.frequent_patterns import apriori, association_rules


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
# EXPERIMENT SETTINGS
# ---------------------------------------------------------

SAMPLE_SIZE = 100_000

MAX_LEN = 3

EXPERIMENTS = [
    {
        "name": "Experiment_A",
        "min_support": 0.01,
        "min_confidence": 0.30
    },
    {
        "name": "Experiment_B",
        "min_support": 0.005,
        "min_confidence": 0.30
    },
    {
        "name": "Experiment_C",
        "min_support": 0.02,
        "min_confidence": 0.30
    },
    {
        "name": "Experiment_D",
        "min_support": 0.01,
        "min_confidence": 0.40
    }
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("APRIORI PARAMETER EXPERIMENTS")
print("=" * 70)

print("\nLoading basket matrix...")

matrix = load_npz(MATRIX_FILE)

print(f"Full matrix: {matrix.shape}")


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
# SELECT SAMPLE
# ---------------------------------------------------------

sample_size = min(
    SAMPLE_SIZE,
    matrix.shape[0]
)

sample_matrix = matrix[:sample_size]


# ---------------------------------------------------------
# CREATE BOOLEAN DATAFRAME
# ---------------------------------------------------------

basket_data = pd.DataFrame.sparse.from_spmatrix(
    sample_matrix,
    columns=product_names
).astype(bool)

print(
    f"\nBaskets used: {len(basket_data):,}"
)

print(
    f"Products: {len(basket_data.columns)}"
)


# ---------------------------------------------------------
# RUN EXPERIMENTS
# ---------------------------------------------------------

results = []

for experiment in EXPERIMENTS:

    name = experiment["name"]
    min_support = experiment["min_support"]
    min_confidence = experiment["min_confidence"]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Minimum support    : {min_support}")
    print(f"Minimum confidence : {min_confidence}")

    # -----------------------------------------------------
    # APRIORI
    # -----------------------------------------------------

    start_time = time.perf_counter()

    frequent_itemsets = apriori(
        basket_data,
        min_support=min_support,
        use_colnames=True,
        max_len=MAX_LEN,
        low_memory=True
    )

    runtime = time.perf_counter() - start_time

    # -----------------------------------------------------
    # ASSOCIATION RULES
    # -----------------------------------------------------

    if not frequent_itemsets.empty:

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        rules = rules[
            rules["lift"] >= 1.0
        ].copy()

    else:

        rules = pd.DataFrame()

    # -----------------------------------------------------
    # RULE STATISTICS
    # -----------------------------------------------------

    if not rules.empty:

        average_confidence = rules[
            "confidence"
        ].mean()

        average_lift = rules[
            "lift"
        ].mean()

        maximum_lift = rules[
            "lift"
        ].max()

    else:

        average_confidence = 0
        average_lift = 0
        maximum_lift = 0

    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    result = {
        "experiment": name,
        "sample_baskets": sample_size,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "frequent_itemsets": len(frequent_itemsets),
        "association_rules": len(rules),
        "runtime_seconds": round(runtime, 4),
        "average_confidence": round(
            average_confidence,
            4
        ),
        "average_lift": round(
            average_lift,
            4
        ),
        "maximum_lift": round(
            maximum_lift,
            4
        )
    }

    results.append(result)

    # -----------------------------------------------------
    # PRINT RESULT
    # -----------------------------------------------------

    print(
        f"\nFrequent itemsets : "
        f"{len(frequent_itemsets):,}"
    )

    print(
        f"Association rules : "
        f"{len(rules):,}"
    )

    print(
        f"Runtime            : "
        f"{runtime:.4f} seconds"
    )

    print(
        f"Average confidence : "
        f"{average_confidence:.4f}"
    )

    print(
        f"Average lift       : "
        f"{average_lift:.4f}"
    )

    print(
        f"Maximum lift       : "
        f"{maximum_lift:.4f}"
    )


# ---------------------------------------------------------
# SAVE EXPERIMENT RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "apriori_experiment_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY FINAL TABLE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved results to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("EXPERIMENTS COMPLETE")
print("=" * 70)