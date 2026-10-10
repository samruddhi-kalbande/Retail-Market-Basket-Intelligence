# Retail Market Basket Intelligence

Retail Market Basket Intelligence analyzes shopping baskets to find products that occur together and turn those patterns into cross-selling recommendations. It compares the Apriori and FP-Growth frequent-pattern mining algorithms and provides an interactive Streamlit dashboard for exploring their results.

**Live dashboard:** [Open the app](https://retail-market-basket-intelligence-q5rhkcpagibcfdfafab6rx.streamlit.app/)

## Dashboard

The dashboard includes:

- **Overview** of basket and product statistics, frequent products, and top association rules.
- **Frequent Itemsets** with product search, support filtering, and CSV download.
- **Association Rules** that can be searched and ranked by support, confidence, or lift.
- **Product Recommendations** based on products selected in a basket.
- **Algorithm Comparison** for reviewing Apriori and FP-Growth results.

The current full-data analysis contains 2,636,756 baskets across 46 product categories. The dashboard filters the saved mining results; changing its settings does not rerun the algorithms.

## Repository contents

```text
app/       Streamlit dashboard
src/       Data preparation, mining, recommendation, and experiment scripts
outputs/   Saved mining results and experiment summaries used by the dashboard
data/      Small project metadata; raw data and large generated files are excluded
```

## Requirements

- Python 3
- Packages listed in [`requirements.txt`](requirements.txt)
- The ARHyMST dataset v1.0 transaction CSV files to regenerate the analysis

Install the dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard

From the project root, start the app:

```powershell
streamlit run app/app.py
```

The app reads the full-data Apriori and FP-Growth results from `outputs/`. Those result files are included in the repository, so the dashboard can be run without downloading the raw dataset.

## Rebuild the analysis

The raw dataset and large processed files are not included in this repository. Obtain ARHyMST dataset v1.0 separately, and place its extracted transaction files here:

```text
data/raw/ARHyMST_dataset_v1.0/Data/transactions_part_01.csv
data/raw/ARHyMST_dataset_v1.0/Data/transactions_part_02.csv
...
```

Keep the `transactions_part_*.csv` naming pattern. Then, from the project root, run the scripts in order:

```powershell
python src/preprocessing.py
python src/one_hot_encoding.py
python src/apriori_analysis.py
python src/fp_growth_analysis.py
```

The first two scripts create the cleaned transaction data, basket summary, and sparse basket matrix under `data/processed/`. The mining scripts save full-data frequent itemsets, association rules, and runtime summaries under `outputs/`, which the dashboard reads. Full-data preprocessing and mining can require substantial memory and compute time.

Optional experiments:

```powershell
python src/apriori_experiments.py
python src/scalability_experiment.py
```

## Mining concepts

- **Support** measures the fraction of baskets containing an itemset.
- **Confidence** measures how often baskets containing the rule's antecedent also contain its consequent.
- **Lift** compares that co-occurrence with what would be expected if the products occurred independently; values above 1 indicate a positive association.

## Data and generated files

The raw dataset and large generated transaction/matrix files are excluded by [`.gitignore`](.gitignore), keeping them out of ordinary Git pushes. The saved CSV mining results in `outputs/` are included so the dashboard can be explored without rerunning the pipeline. Check the dataset's terms before obtaining or redistributing it.
