from pathlib import Path
import time

from scipy.sparse import load_npz
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth, association_rules


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATRIX_FILE = PROJECT_ROOT / "data" / "processed" / "basket_matrix.npz"
PRODUCTS_FILE = PROJECT_ROOT / "data" / "processed" / "product_categories.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PARAMETERS — SAME AS APRIORI
# --------------------------------------------------

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0
MAX_LEN = 3


# --------------------------------------------------
# LOAD FULL DATASET
# --------------------------------------------------

print("=" * 60)
print("FP-GROWTH — FULL DATASET ANALYSIS")
print("=" * 60)

print("\nLoading complete basket matrix...")

matrix = load_npz(MATRIX_FILE)

with open(PRODUCTS_FILE, "r", encoding="utf-8") as file:
    product_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

print(f"Total baskets: {matrix.shape[0]:,}")
print(f"Product categories: {matrix.shape[1]}")

if matrix.shape[1] != len(product_names):
    raise ValueError(
        "The number of product names does not match the matrix columns."
    )

basket_data = pd.DataFrame.sparse.from_spmatrix(
    matrix,
    columns=product_names
).astype(bool)


# --------------------------------------------------
# RUN FP-GROWTH
# --------------------------------------------------

print("\nStarting FP-Growth...")
print(f"Minimum support: {MIN_SUPPORT}")
print(f"Maximum itemset length: {MAX_LEN}")

start_time = time.perf_counter()

frequent_itemsets = fpgrowth(
    basket_data,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_LEN
)

runtime = time.perf_counter() - start_time

print(f"\nFrequent itemsets found: {len(frequent_itemsets):,}")
print(f"FP-Growth runtime: {runtime:.2f} seconds")


# --------------------------------------------------
# GENERATE ASSOCIATION RULES
# --------------------------------------------------

print("\nGenerating association rules...")

if frequent_itemsets.empty:
    rules = pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift"
        ]
    )
else:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    rules = rules[
        rules["lift"] >= MIN_LIFT
    ].copy()

print(f"Association rules found: {len(rules):,}")


# --------------------------------------------------
# SAVE SEPARATE FULL-DATA RESULTS
# --------------------------------------------------

itemsets_path = OUTPUT_DIR / "fpgrowth_frequent_itemsets_full.csv"
rules_path = OUTPUT_DIR / "fpgrowth_association_rules_full.csv"
runtime_path = OUTPUT_DIR / "fpgrowth_full_runtime.txt"

frequent_itemsets.to_csv(itemsets_path, index=False)
rules.to_csv(rules_path, index=False)

with open(runtime_path, "w", encoding="utf-8") as file:
    file.write("Algorithm: FP-Growth\n")
    file.write(f"Number of baskets: {matrix.shape[0]}\n")
    file.write(f"Number of product categories: {matrix.shape[1]}\n")
    file.write(f"Minimum support: {MIN_SUPPORT}\n")
    file.write(f"Minimum confidence: {MIN_CONFIDENCE}\n")
    file.write(f"Minimum lift: {MIN_LIFT}\n")
    file.write(f"Maximum itemset length: {MAX_LEN}\n")
    file.write(f"Frequent itemsets: {len(frequent_itemsets)}\n")
    file.write(f"Association rules: {len(rules)}\n")
    file.write(f"Runtime seconds: {runtime:.2f}\n")

print("\nFULL-DATA FP-GROWTH COMPLETED")
print(f"Itemsets saved to: {itemsets_path}")
print(f"Rules saved to: {rules_path}")
print(f"Runtime saved to: {runtime_path}")

if not rules.empty:
    print("\nTop 10 rules by lift:")
    print(
        rules.sort_values(
            ["lift", "confidence"],
            ascending=False
        ).head(10)[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].to_string(index=False)
    )

print("\nYour existing sample results have not been changed.")
