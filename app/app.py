
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Retail Intelligence | Market Basket Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        background: linear-gradient(115deg, #12372A, #245C45);
        color: white;
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #DCECE3;
        margin-bottom: 0;
    }

    .section-note {
        color: #7B8794;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.23);
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 8px;
    }

    h2, h3 {
        letter-spacing: -0.3px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_results():
    files = {
        "apriori_itemsets": "apriori_frequent_itemsets_full.csv",
        "apriori_rules": "apriori_association_rules_full.csv",
        "fp_itemsets": "fpgrowth_frequent_itemsets_full.csv",
        "fp_rules": "fpgrowth_association_rules_full.csv",
    }

    data = {}

    for key, filename in files.items():
        path = OUTPUTS / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {filename}. Run the full-data mining script first."
            )

        data[key] = pd.read_csv(path)

    return data


try:
    data = load_results()
except Exception as error:
    st.error("Unable to load the full-dataset mining results.")
    st.exception(error)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_itemset(value):
    """Read product names from saved frozenset strings."""
    if pd.isna(value):
        return set()

    return set(re.findall(r"'([^']*)'", str(value)))


def pretty_itemset(value):
    items = sorted(parse_itemset(value))
    return " + ".join(items) if items else str(value)


def make_recommendations(rules, selected, top_n=10):
    """
    Rank recommendations by the number of selected products
    matched in the rule antecedent, then lift and confidence.
    """
    selected = set(selected)
    candidates = []

    for _, rule in rules.iterrows():
        antecedents = parse_itemset(rule["antecedents"])
        consequents = parse_itemset(rule["consequents"])

        if not antecedents or not consequents:
            continue

        # Only use rules whose complete antecedent is in the basket.
        if antecedents.issubset(selected):
            for product in consequents - selected:
                candidates.append({
                    "Recommended Product": product,
                    "Matched Products": pretty_itemset(
                        rule["antecedents"]
                    ),
                    "Support": float(rule["support"]),
                    "Confidence": float(rule["confidence"]),
                    "Lift": float(rule["lift"]),
                    "Matched Antecedent Size": len(antecedents),
                })

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates)

    result = result.sort_values(
        ["Matched Antecedent Size", "Lift", "Confidence"],
        ascending=[False, False, False],
    )

    result = result.drop_duplicates(
        subset=["Recommended Product"],
        keep="first",
    )

    return result.head(top_n).drop(
        columns=["Matched Antecedent Size"]
    ).reset_index(drop=True)


def show_table(df, height=420):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛒 Retail Intelligence")
    st.caption("MARKET BASKET ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Frequent Itemsets",
            "Association Rules",
            "Product Recommendations",
            "Algorithm Comparison",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("### Analysis settings")

    algorithm = st.selectbox(
        "Mining algorithm",
        ["Apriori", "FP-Growth"],
    )

    min_support = st.slider(
        "Minimum support",
        0.001, 0.10, 0.01, 0.001,
        format="%.3f",
    )

    min_confidence = st.slider(
        "Minimum confidence",
        0.10, 1.00, 0.30, 0.05,
        format="%.2f",
    )

    min_lift = st.slider(
        "Minimum lift",
        1.0, 5.0, 1.0, 0.1,
        format="%.1f",
    )

    st.caption(
        "Filters apply to saved full-data results. "
        "They do not rerun the mining algorithms."
    )

    if st.button("Reset filters", use_container_width=True):
        st.session_state["support_reset"] = True
        st.rerun()


# Reset slider values through widget keys on the next rerun.
if st.session_state.pop("support_reset", False):
    for key in ["support_filter", "confidence_filter", "lift_filter"]:
        st.session_state.pop(key, None)


# ============================================================
# SELECT CURRENT ALGORITHM RESULTS
# ============================================================

if algorithm == "Apriori":
    itemsets = data["apriori_itemsets"].copy()
    rules = data["apriori_rules"].copy()
    runtime = 12.76
else:
    itemsets = data["fp_itemsets"].copy()
    rules = data["fp_rules"].copy()
    runtime = 35.54


filtered_itemsets = itemsets[
    itemsets["support"] >= min_support
].copy()

filtered_rules = rules[
    (rules["support"] >= min_support)
    & (rules["confidence"] >= min_confidence)
    & (rules["lift"] >= min_lift)
].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>Retail Market Basket Intelligence</h1>
    <p>Discover product associations, evaluate mining algorithms,
    and generate data-driven cross-selling recommendations.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Active algorithm: {algorithm}  ·  "
    "Analysis uses the complete basket matrix"
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.subheader("Business overview")
    st.caption(
        "Dataset statistics and mining results from the full basket dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transaction baskets", "2.64M")
    c2.metric("Unique customers", "61,298")
    c3.metric("Product categories", "46")
    c4.metric("Sales channels", "2")

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Frequent itemsets",
        f"{len(filtered_itemsets):,}",
    )

    c2.metric(
        "Association rules",
        f"{len(filtered_rules):,}",
    )

    c3.metric(
        "Mining runtime",
        f"{runtime:.2f} sec",
    )

    st.subheader("Most frequent product categories")

    single_items = filtered_itemsets[
        filtered_itemsets["itemsets"].apply(
            lambda value: len(parse_itemset(value)) == 1
        )
    ].copy()

    if not single_items.empty:
        single_items["Product"] = single_items["itemsets"].apply(
            pretty_itemset
        )

        top_products = single_items.nlargest(15, "support")

        fig = px.bar(
            top_products.sort_values("support"),
            x="support",
            y="Product",
            orientation="h",
            text=top_products.sort_values("support")["support"].map(
                lambda x: f"{x:.1%}"
            ),
            labels={"support": "Support", "Product": ""},
            title="Product category prevalence",
            color="support",
            color_continuous_scale=["#B7D8C5", "#176B4D"],
        )

        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=5, r=10, t=50, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top association rules")

    if not filtered_rules.empty:
        top_rules = filtered_rules.nlargest(10, "lift").copy()

        top_rules["Rule"] = (
            top_rules["antecedents"].apply(pretty_itemset)
            + " → "
            + top_rules["consequents"].apply(pretty_itemset)
        )

        show_table(
            top_rules[
                ["Rule", "support", "confidence", "lift"]
            ].rename(columns={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            }),
            height=360,
        )
    else:
        st.info("No rules match the current filters.")


# ============================================================
# FREQUENT ITEMSETS PAGE
# ============================================================

elif page == "Frequent Itemsets":

    st.subheader("📦 Frequent itemsets")
    st.write(
        "Itemsets are product categories that occur together "
        "at or above the selected support threshold."
    )

    search = st.text_input(
        "Search product or itemset",
        placeholder="e.g. CHEESE or POULTRY",
    )

    result = filtered_itemsets.copy()
    result["Products"] = result["itemsets"].apply(pretty_itemset)
    result["Number of products"] = result["itemsets"].apply(
        lambda value: len(parse_itemset(value))
    )

    if search:
        result = result[
            result["Products"].str.contains(
                search, case=False, na=False
            )
        ]

    result = result.sort_values("support", ascending=False)

    c1, c2 = st.columns(2)
    c1.metric("Matching itemsets", f"{len(result):,}")
    c2.metric(
        "Largest itemset",
        int(result["Number of products"].max()) if not result.empty else 0,
    )

    show_table(
        result[
            ["Products", "Number of products", "support"]
        ].rename(columns={"support": "Support"}),
        height=500,
    )

    st.download_button(
        "Download filtered itemsets CSV",
        data=result.to_csv(index=False),
        file_name=f"{algorithm.lower()}_full_itemsets.csv",
        mime="text/csv",
    )


# ============================================================
# ASSOCIATION RULES PAGE
# ============================================================

elif page == "Association Rules":

    st.subheader("🔗 Association rule explorer")
    st.write(
        "Explore directional relationships between antecedent products "
        "and consequent products."
    )

    search = st.text_input(
        "Search rules",
        placeholder="Search for CHEESE, POULTRY, DAIRY...",
    )

    sort_by = st.selectbox(
        "Rank rules by",
        ["lift", "confidence", "support"],
        index=0,
    )

    result = filtered_rules.copy()

    result["If basket contains"] = result["antecedents"].apply(
        pretty_itemset
    )
    result["Then recommend"] = result["consequents"].apply(
        pretty_itemset
    )

    if search:
        mask = (
            result["If basket contains"].str.contains(
                search, case=False, na=False
            )
            | result["Then recommend"].str.contains(
                search, case=False, na=False
            )
        )
        result = result[mask]

    result = result.sort_values(sort_by, ascending=False)

    st.metric("Rules matching filters", f"{len(result):,}")

    show_table(
        result[
            [
                "If basket contains",
                "Then recommend",
                "support",
                "confidence",
                "lift",
            ]
        ].rename(columns={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }),
        height=500,
    )

    if not result.empty:
        fig = px.scatter(
            result.head(500),
            x="support",
            y="confidence",
            size="lift",
            hover_data=[
                "If basket contains",
                "Then recommend",
                "lift",
            ],
            title="Rule support versus confidence",
            labels={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            },
            color="lift",
            color_continuous_scale=["#B7D8C5", "#176B4D"],
        )

        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download filtered rules CSV",
        data=result.to_csv(index=False),
        file_name=f"{algorithm.lower()}_full_rules.csv",
        mime="text/csv",
    )


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

elif page == "Product Recommendations":

    st.subheader("💡 Product recommendation studio")

    st.write(
        "Choose the products already in a basket. The system searches "
        "association rules whose antecedents match those products."
    )

    product_file = PROCESSED / "product_categories.txt"

    try:
        products = [
            line.strip()
            for line in open(product_file, encoding="utf-8")
            if line.strip()
        ]
    except FileNotFoundError:
        st.error("Product categories file not found.")
        st.stop()

    selected = st.multiselect(
        "Products currently in the basket",
        options=products,
        placeholder="Select one or more product categories",
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )

    if st.button("Generate recommendations", type="primary"):
        if not selected:
            st.warning("Select at least one product first.")
        else:
            recommendations = make_recommendations(
                filtered_rules,
                selected,
                top_n=top_n,
            )

            if recommendations.empty:
                st.warning(
                    "No rules match this basket and your current filters. "
                    "Try lowering confidence or support, or select other products."
                )
            else:
                st.success(
                    f"Found {len(recommendations)} product recommendations."
                )

                show_table(recommendations, height=450)

                st.download_button(
                    "Download recommendations",
                    data=recommendations.to_csv(index=False),
                    file_name="product_recommendations.csv",
                    mime="text/csv",
                )

    st.info(
        "Recommendations show associations, not causal effects or guaranteed purchases."
    )


# ============================================================
# ALGORITHM COMPARISON PAGE
# ============================================================

elif page == "Algorithm Comparison":

    st.subheader("⚖️ Apriori vs FP-Growth")

    comparison = pd.DataFrame({
        "Algorithm": ["Apriori", "FP-Growth"],
        "Runtime (seconds)": [12.76, 35.54],
        "Frequent itemsets": [
            len(data["apriori_itemsets"]),
            len(data["fp_itemsets"]),
        ],
        "Association rules": [
            len(data["apriori_rules"]),
            len(data["fp_rules"]),
        ],
    })

    c1, c2 = st.columns(2)

    c1.metric("Apriori runtime", "12.76 sec")
    c2.metric("FP-Growth runtime", "35.54 sec")

    st.plotly_chart(
        px.bar(
            comparison,
            x="Algorithm",
            y="Runtime (seconds)",
            color="Algorithm",
            title="Execution time on the full dataset",
            color_discrete_map={
                "Apriori": "#176B4D",
                "FP-Growth": "#8AB7A0",
            },
        ).update_layout(showlegend=False),
        use_container_width=True,
    )

    st.subheader("Measured results")
    show_table(comparison, height=180)

    st.markdown("""
    **Experimental conclusion**

    Under the tested conditions, Apriori completed faster than FP-Growth.
    Both algorithms discovered the same number of frequent itemsets and
    association rules using the same support, confidence, lift and maximum
    itemset-length settings.

    Execution time depends on the dataset, implementation, parameters and
    computer hardware; these results should not be generalized to every dataset.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Retail Market Basket Intelligence · Full-dataset association analysis "
    "· Apriori and FP-Growth"
)