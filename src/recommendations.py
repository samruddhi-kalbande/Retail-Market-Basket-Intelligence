from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RULES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "apriori_association_rules.csv"
)


# ---------------------------------------------------------
# LOAD RULES
# ---------------------------------------------------------

print("=" * 70)
print("RETAIL PRODUCT RECOMMENDATION ENGINE")
print("=" * 70)

print("\nLoading association rules...")

rules = pd.read_csv(RULES_FILE)

print(f"Rules loaded: {len(rules):,}")


# ---------------------------------------------------------
# CONVERT TEXT BACK TO PRODUCT SETS
# ---------------------------------------------------------

def parse_product_set(text):
    """
    Convert a saved frozenset string into a Python set.
    """

    text = str(text)

    text = text.replace(
        "frozenset({",
        ""
    )

    text = text.replace(
        "})",
        ""
    )

    text = text.strip()

    if not text:
        return set()

    products = [
        product.strip().strip("'\"")
        for product in text.split(",")
    ]

    return set(products)


rules["antecedents_set"] = (
    rules["antecedents"]
    .apply(parse_product_set)
)

rules["consequents_set"] = (
    rules["consequents"]
    .apply(parse_product_set)
)


# ---------------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------------

def recommend_products(
    selected_products,
    top_n=10
):
    """
    Recommend products based on association rules.

    Parameters
    ----------
    selected_products : list
        Products already selected by the user.

    top_n : int
        Number of recommendations to return.

    Returns
    -------
    DataFrame
        Ranked recommendations.
    """

    selected_products = set(
        selected_products
    )

    matching_rules = []

    for _, rule in rules.iterrows():

        antecedent = rule[
            "antecedents_set"
        ]

        consequent = rule[
            "consequents_set"
        ]

        # The rule applies when all
        # antecedent products are selected.
        if antecedent.issubset(
            selected_products
        ):

            # Don't recommend a product
            # that the user already selected.
            new_products = (
                consequent - selected_products
            )

            for product in new_products:

                matching_rules.append(
                    {
                        "recommended_product":
                            product,

                        "confidence":
                            rule["confidence"],

                        "lift":
                            rule["lift"],

                        "support":
                            rule["support"],

                        "antecedent":
                            ", ".join(
                                sorted(antecedent)
                            )
                    }
                )


    # -----------------------------------------------------
    # NO RECOMMENDATIONS
    # -----------------------------------------------------

    if not matching_rules:

        return pd.DataFrame(
            columns=[
                "recommended_product",
                "confidence",
                "lift",
                "support",
                "antecedent"
            ]
        )


    recommendations = pd.DataFrame(
        matching_rules
    )


    # -----------------------------------------------------
    # REMOVE DUPLICATE PRODUCTS
    # -----------------------------------------------------

    recommendations = (
        recommendations
        .sort_values(
            ["lift", "confidence"],
            ascending=False
        )
        .drop_duplicates(
            subset=["recommended_product"]
        )
        .head(top_n)
        .reset_index(drop=True)
    )


    return recommendations


# ---------------------------------------------------------
# TEST RECOMMENDATION
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TEST RECOMMENDATION")
print("-" * 70)

selected_products = [
    "CHEESE"
]

print(
    "\nSelected product:"
)

print(selected_products)


recommendations = recommend_products(
    selected_products,
    top_n=10
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if recommendations.empty:

    print(
        "\nNo recommendations found."
    )

else:

    print(
        "\nRecommended products:"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# SECOND TEST
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("SECOND TEST")
print("-" * 70)

selected_products = [
    "CHEESE",
    "POULTRY"
]

print(
    "\nSelected products:"
)

print(selected_products)


recommendations = recommend_products(
    selected_products,
    top_n=10
)


if recommendations.empty:

    print(
        "\nNo recommendations found."
    )

else:

    print(
        "\nRecommended products:"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RECOMMENDATION ENGINE TEST COMPLETE")
print("=" * 70)