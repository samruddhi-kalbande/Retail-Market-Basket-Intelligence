import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


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

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


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

    loaded = {}

    for key, filename in files.items():
        path = OUTPUTS / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {filename} at {path}. "
                "Run the full-data mining script and ensure output files are committed."
            )
        loaded[key] = pd.read_csv(path)

    return loaded


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
    """Parse item names from saved frozenset/list-like strings."""
    if pd.isna(value):
        return set()

    text = str(value)
    # Typical saved format: frozenset({'CHEESE', 'POULTRY'})
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if quoted_items:
        return set(quoted_items)

    # Fallback for a simple unquoted item string.
    cleaned = re.sub(r"^(frozenset|set)\(", "", text).rstrip(")}]")
    return {part.strip() for part in cleaned.split(",") if part.strip()}


def pretty_itemset(value):
    items = sorted(parse_itemset(value))
    return " + ".join(items) if items else str(value)


def make_recommendations(rules, selected, top_n=10):
    """Recommend consequents when the complete antecedent is selected."""
    selected = set(selected)
    candidates = []

    for _, rule in rules.iterrows():
        antecedents = parse_itemset(rule["antecedents"])
        consequents = parse_itemset(rule["consequents"])

        if not antecedents or not consequents:
            continue

        if antecedents.issubset(selected):
            for product in consequents - selected:
                candidates.append(
                    {
                        "Recommended Category": product,
                        "Matched Categories": pretty_itemset(rule["antecedents"]),
                        "Support": float(rule["support"]),
                        "Confidence": float(rule["confidence"]),
                        "Lift": float(rule["lift"]),
                        "Matched Antecedent Size": len(antecedents),
                    }
                )

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates)
    result = result.sort_values(
        ["Matched Antecedent Size", "Lift", "Confidence"],
        ascending=[False, False, False],
    )
    result = result.drop_duplicates(
        subset=["Recommended Category"],
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


def rule_display_frame(rules):
    """Create readable rule columns without changing source data."""
    result = rules.copy()
    result["If basket contains"] = result["antecedents"].apply(pretty_itemset)
    result["Then recommend"] = result["consequents"].apply(pretty_itemset)
    return result


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
            "Business Insights",
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
        min_value=0.001,
        max_value=0.10,
        value=0.01,
        step=0.001,
        format="%.3f",
    )

    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=1.00,
        value=0.30,
        step=0.05,
        format="%.2f",
    )

    min_lift = st.slider(
        "Minimum lift",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )

    st.caption(
        "Filters apply to saved full-data results. "
        "They do not rerun the mining algorithms."
    )

    if st.button("Reset filters", use_container_width=True):
        st.session_state["min_support"] = 0.01
        st.session_state["min_confidence"] = 0.30
        st.session_state["min_lift"] = 1.0
        st.rerun()


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

# Ensure expected numeric columns are numeric, even if CSV inference differs.
for frame, columns in [
    (itemsets, ["support"]),
    (rules, ["support", "confidence", "lift"]),
]:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

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

st.markdown(
    """
    <div class="hero">
        <h1>Retail Market Basket Intelligence</h1>
        <p>Discover product associations, evaluate mining algorithms,
        and generate data-driven cross-selling recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Active algorithm: {algorithm} · "
    "Analysis uses saved results from the complete basket dataset"
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
    c1.metric("Frequent itemsets", f"{len(filtered_itemsets):,}")
    c2.metric("Association rules", f"{len(filtered_rules):,}")
    c3.metric("Recorded mining runtime", f"{runtime:.2f} sec")

    st.subheader("Most frequent product categories")
    single_items = filtered_itemsets[
        filtered_itemsets["itemsets"].apply(
            lambda value: len(parse_itemset(value)) == 1
        )
    ].copy()

    if not single_items.empty:
        single_items["Product category"] = single_items["itemsets"].apply(
            pretty_itemset
        )
        top_products = single_items.nlargest(15, "support").sort_values("support")

        fig = px.bar(
            top_products,
            x="support",
            y="Product category",
            orientation="h",
            text=top_products["support"].map(lambda value: f"{value:.1%}"),
            labels={"support": "Support", "Product category": ""},
            title="Product category prevalence",
            color="support",
            color_continuous_scale=["#B7D8C5", "#176B4D"],
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=5, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No single-item itemsets match the current support filter.")

    st.subheader("Top association rules")
    if not filtered_rules.empty:
        top_rules = filtered_rules.nlargest(10, "lift").copy()
        top_rules["Rule"] = (
            top_rules["antecedents"].apply(pretty_itemset)
            + " → "
            + top_rules["consequents"].apply(pretty_itemset)
        )
        show_table(
            top_rules[["Rule", "support", "confidence", "lift"]].rename(
                columns={
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }
            ),
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
        key="itemset_search",
    )

    result = filtered_itemsets.copy()
    result["Products"] = result["itemsets"].apply(pretty_itemset)
    result["Number of categories"] = result["itemsets"].apply(
        lambda value: len(parse_itemset(value))
    )

    if search:
        result = result[
            result["Products"].str.contains(search, case=False, na=False, regex=False)
        ]

    result = result.sort_values("support", ascending=False)

    c1, c2 = st.columns(2)
    c1.metric("Matching itemsets", f"{len(result):,}")
    c2.metric(
        "Largest itemset",
        int(result["Number of categories"].max()) if not result.empty else 0,
    )

    show_table(
        result[["Products", "Number of categories", "support"]].rename(
            columns={"support": "Support"}
        ),
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
        "Explore directional relationships between antecedent categories "
        "and consequent categories."
    )

    search = st.text_input(
        "Search rules",
        placeholder="Search for CHEESE, POULTRY, DAIRY...",
        key="rules_search",
    )

    sort_by = st.selectbox(
        "Rank rules by",
        ["lift", "confidence", "support"],
        index=0,
    )

    result = rule_display_frame(filtered_rules)

    if search:
        mask = (
            result["If basket contains"].str.contains(
                search, case=False, na=False, regex=False
            )
            | result["Then recommend"].str.contains(
                search, case=False, na=False, regex=False
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
        ].rename(
            columns={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            }
        ),
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
# PRODUCT RECOMMENDATIONS PAGE
# ============================================================

elif page == "Product Recommendations":
    st.subheader("💡 Product recommendation studio")
    st.write(
        "Choose categories already in a basket. The system searches "
        "association rules whose antecedents match those categories."
    )

    product_file = PROCESSED / "product_categories.txt"

    try:
        with open(product_file, encoding="utf-8") as file:
            products = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        st.error(f"Product categories file not found: {product_file}")
        st.stop()

    selected = st.multiselect(
        "Categories currently in the basket",
        options=products,
        placeholder="Select one or more product categories",
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
        key="recommendation_count",
    )

    if st.button("Generate recommendations", type="primary"):
        if not selected:
            st.warning("Select at least one category first.")
        else:
            recommendations = make_recommendations(
                filtered_rules,
                selected,
                top_n=top_n,
            )

            if recommendations.empty:
                st.warning(
                    "No rules match this basket and the current filters. "
                    "Try lowering confidence, support, or lift, or select other categories."
                )
            else:
                st.success(
                    f"Found {len(recommendations)} category recommendations."
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

    comparison = pd.DataFrame(
        {
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
        }
    )

    c1, c2 = st.columns(2)
    c1.metric("Apriori runtime", "12.76 sec")
    c2.metric("FP-Growth runtime", "35.54 sec")

    fig = px.bar(
        comparison,
        x="Algorithm",
        y="Runtime (seconds)",
        color="Algorithm",
        title="Execution time on the full dataset",
        color_discrete_map={
            "Apriori": "#176B4D",
            "FP-Growth": "#8AB7A0",
        },
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Measured results")
    show_table(comparison, height=180)

    st.markdown(
        """
        **Experimental conclusion**

        Under the tested conditions, Apriori completed faster than FP-Growth.
        Both algorithms discovered the same number of frequent itemsets and
        association rules using the same mining settings.

        Execution time depends on the dataset, implementation, parameters,
        and computer hardware; these results should not be generalized to
        every dataset.
        """
    )


# ============================================================
# BUSINESS INSIGHTS PAGE
# ============================================================

elif page == "Business Insights":
    st.subheader("💼 Business Insights")
    st.write(
        "Turn association rules into potential retail actions for product "
        "bundling, shelf placement, and promotional combinations."
    )

    st.info(
        "These suggestions are based on category-level associations. "
        "They do not prove that a placement change or promotion will increase sales."
    )

    bundle_tab, shelf_tab, promo_tab = st.tabs(
        ["📦 Product Bundles", "🛒 Shelf Placement", "🏷️ Promotional Combinations"]
    )

    # --------------------------------------------------------
    # PRODUCT BUNDLES
    # --------------------------------------------------------
    with bundle_tab:
        st.markdown("### Product bundle suggestions")
        st.write(
            "Rank potential category bundles using association-rule metrics."
        )

        bundle_lift = st.slider(
            "Minimum lift for bundles",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="business_bundle_lift",
        )

        bundles = filtered_rules[
            filtered_rules["lift"] >= bundle_lift
        ].copy()

        if not bundles.empty:
            bundles["Suggested bundle"] = (
                bundles["antecedents"].apply(pretty_itemset)
                + " + "
                + bundles["consequents"].apply(pretty_itemset)
            )

            bundles = bundles.sort_values(
                ["lift", "confidence", "support"],
                ascending=False,
            ).drop_duplicates(subset=["Suggested bundle"])

            bundle_display = bundles[
                ["Suggested bundle", "support", "confidence", "lift"]
            ].rename(
                columns={
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Bundle candidates", len(bundle_display))
            c2.metric("Highest lift", f"{bundle_display['Lift'].max():.2f}")
            c3.metric("Highest confidence", f"{bundle_display['Confidence'].max():.1%}")

            show_table(bundle_display.head(30), height=420)

            st.download_button(
                "Download bundle suggestions",
                data=bundle_display.to_csv(index=False),
                file_name="product_bundle_suggestions.csv",
                mime="text/csv",
                key="business_download_bundles",
            )
        else:
            st.warning(
                "No bundle candidates match this threshold. Lower the lift "
                "threshold or adjust the sidebar filters."
            )

    # --------------------------------------------------------
    # SHELF PLACEMENT
    # --------------------------------------------------------
    with shelf_tab:
        st.markdown("### Shelf-placement suggestions")
        st.write(
            "Find pairs of single categories associated with one another. "
            "These pairs may be considered for nearby placement."
        )

        shelf_rules = filtered_rules[
            filtered_rules["antecedents"].apply(
                lambda value: len(parse_itemset(value)) == 1
            )
            & filtered_rules["consequents"].apply(
                lambda value: len(parse_itemset(value)) == 1
            )
            & (filtered_rules["lift"] > 1)
        ].copy()

        pair_records = []

        for _, rule in shelf_rules.iterrows():
            left_items = parse_itemset(rule["antecedents"])
            right_items = parse_itemset(rule["consequents"])

            if len(left_items) != 1 or len(right_items) != 1:
                continue

            left = next(iter(left_items))
            right = next(iter(right_items))

            if left == right:
                continue

            category_a, category_b = sorted([left, right])
            pair_records.append(
                {
                    "Category A": category_a,
                    "Category B": category_b,
                    "Support": float(rule["support"]),
                    "Confidence": float(rule["confidence"]),
                    "Lift": float(rule["lift"]),
                }
            )

        if pair_records:
            shelf_df = pd.DataFrame(pair_records)
            shelf_df = (
                shelf_df.sort_values(
                    ["Lift", "Confidence", "Support"],
                    ascending=False,
                )
                .drop_duplicates(subset=["Category A", "Category B"])
                .reset_index(drop=True)
            )

            st.metric("Potential category pairs", len(shelf_df))
            show_table(shelf_df.head(30), height=420)

            st.download_button(
                "Download shelf-placement suggestions",
                data=shelf_df.to_csv(index=False),
                file_name="shelf_placement_suggestions.csv",
                mime="text/csv",
                key="business_download_shelf",
            )
        else:
            st.warning(
                "No single-category pairs match the current filters. "
                "Try lowering the sidebar support or confidence thresholds."
            )

        st.caption(
            "Validate placement ideas in a real store or controlled experiment "
            "before treating them as effective."
        )

    # --------------------------------------------------------
    # PROMOTIONAL COMBINATIONS
    # --------------------------------------------------------
    with promo_tab:
        st.markdown("### Promotional combination ideas")
        st.write(
            "Identify associated categories that may be candidates for "
            "joint offers or cross-category promotions."
        )

        promo_confidence = st.slider(
            "Minimum confidence for promotions",
            min_value=0.10,
            max_value=1.00,
            value=0.30,
            step=0.05,
            key="business_promo_confidence",
        )

        promo_lift = st.slider(
            "Minimum lift for promotions",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="business_promo_lift",
        )

        promotions = filtered_rules[
            (filtered_rules["confidence"] >= promo_confidence)
            & (filtered_rules["lift"] >= promo_lift)
        ].copy()

        if not promotions.empty:
            promotions["Basket contains"] = promotions["antecedents"].apply(
                pretty_itemset
            )
            promotions["Potential offer category"] = promotions[
                "consequents"
            ].apply(pretty_itemset)

            promotions["Promotion idea"] = (
                "Consider a joint offer: "
                + promotions["Basket contains"]
                + " + "
                + promotions["Potential offer category"]
            )

            promotions = promotions.sort_values(
                ["lift", "confidence", "support"],
                ascending=False,
            )

            promo_display = promotions[
                [
                    "Promotion idea",
                    "support",
                    "confidence",
                    "lift",
                ]
            ].rename(
                columns={
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }
            )

            c1, c2 = st.columns(2)
            c1.metric("Promotion candidates", len(promo_display))
            c2.metric("Highest lift", f"{promo_display['Lift'].max():.2f}")

            show_table(promo_display.head(30), height=420)

            st.download_button(
                "Download promotion ideas",
                data=promo_display.to_csv(index=False),
                file_name="promotional_combination_ideas.csv",
                mime="text/csv",
                key="business_download_promotions",
            )
        else:
            st.warning(
                "No rules match these promotion thresholds. Lower the thresholds "
                "or adjust the sidebar filters."
            )

        st.caption(
            "These are candidates for testing, not evidence that discounts "
            "will be profitable or increase sales."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Retail Market Basket Intelligence · Full-dataset association analysis "
    "· Apriori and FP-Growth"
)

