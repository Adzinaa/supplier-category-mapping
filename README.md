# Supplier Category Mapping System

Maps each of the supplier's 3,030 category values (2,678 distinct after the supplier's own
"ignore case and spaces" rule) to one of our 150 categories, judging from the products inside.

## Run

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the data
# Ensure the four input CSVs (our_categories.csv, supplier_products.csv, labelled_products.csv, sales.csv)
# are placed in the same directory as the notebook.

# 3. Run the solution
# Open and run all cells in the Jupyter Notebook:
jupyter notebook exploration_and_mapping.ipynb

```

The notebook executes the entire pipeline sequentially (training, classification, aggregation, evaluation) and outputs two files in the same directory:

* `mapping.csv`: The required deliverable with the header `supplier_category,our_category_id`, one line per distinct raw value. Values that differ only by case/spacing get the same answer.

* `weak_spots.csv`: Adds confidence, product count, and revenue at risk per category to identify where the mapping needs human review.

## Evidence it works

Labels of whole categories are hidden, the full pipeline is rebuilt, and the result is scored on the hidden
labels (`GroupKFold` evaluation). This mimics the \~40% of categories (1,709 of 2,634 normalized categories) that have
no hand label at all.

| Metric | Result | 
| ----- | ----- | 
| Hidden labelled products placed correctly | **82.1%** | 
| Best any one-category-per-group mapping could do on the same products | **94.1%** | 
| Groups with 3+ hidden labels that match the majority label | **86.6%** (97 groups) | 

The confidence score is honest, so it tells you what to review:

| confidence bucket | categories | correct | 
| ----- | ----- | ----- | 
| 0.95 – 1.00 | 11 | 100.0% | 
| 0.85 – 0.95 | 14 | 100.0% | 
| 0.70 – 0.85 | 19 | 100.0% | 
| 0.50 – 0.70 | 18 | 88.9% | 
| 0.00 – 0.50 | 35 | 68.6% | 

## What I did

1. **Normalise:** Names exactly as the supplier does (casefold, strip all whitespace) → one answer per group.

2. **Classify every product:** From its name (word + character TF-IDF, linear SVM) trained on the hand labels.
   Brand and the cryptic category name were tested and did not help.

3. **Vote per group:** Sum the product evidence, pick the best single category. Hand labels count as certain.

4. **Retail vs catering:** Our tree has 5 pairs ("… catering packs, 5 kg/l and over" vs "… retail packs"). The model
   works on the merged family; the variant is decided by the parsed pack sizes (retail ≤ 3 kg/l, catering ≥ 5 kg/l). Count-only packs (`x6`, `12pk`) carry no size and do not vote.

Cost: Training is one linear SVM on the labeled short texts. Group aggregation is done efficiently via pandas/numpy native operations. The entire notebook runs in under 15 seconds.

## Where it is weakest, and which mistakes cost most

* **Many supplier categories are genuinely mixed.** 74 of the groups with labels have conflicting labels (the team put products of the same supplier category into different categories of ours). A one-category-per-group mapping cannot be right for them, whatever the model.

* **Money is concentrated in the weak spots.** Groups with confidence < 0.70 hold roughly 50% of the total revenue (1,045 groups) because the big groups are the mixed ones. `weak_spots.csv` ranks all groups by `revenue_at_risk`. The top ones are `pcklscndmn` (EUR \~389k at risk), `citr.frt` (EUR \~386k), and `soft-squashgrp3` (EUR \~372k). Review these first.

* **Retail + catering mixes:** 29 groups contain both pack classes in balanced proportions (10% - 90%); the assigned id is right for only one of the variants.

* **No labels:** 1,709 groups rest on the product model alone.

## What I deliberately did not do

* **Weight votes by revenue.** First version did; it let a few best-sellers hijack a group (e.g., a bestseller lemon forcing a mixed berries/citrus category entirely into citrus). Counting products matched the group's true diversity better. Revenue is used to rank risk, not to choose the category.

* **Deep Learning/LLMs:** Titles are very short. Complex models would drastically increase runtime and dependencies without clear accuracy gains over a well-tuned Linear SVM.

* **No attempt to split groups:** The brief fixes one category per supplier category.

* **No hand fixing of any category.**

## AI use

AI tools (Claude / Gemini) were used for assistance in structuring the code (e.g., regex pattern for pack sizes and vectorized TF-IDF setup). Checked by hand: the aggregation logic, the cross-validation setup, and the financial risk impact metrics.

## Found outside the task, not fixed

* 6 of our categories receive no supplier category (Lemonade & mixers, Jams & sweet spreads,
  Stock & gravy, Burgers & nuggets, Bananas, Fresh fish). A range gap, or a category tree that has more than this supplier needs.

* The supplier files misplaced products inside its own categories. Moving "category by category" will
  carry that mess into our tree. A per-product check on the low-confidence groups (from `weak_spots.csv`) would be worth the effort.