# Analyzing-Mental-Health-A-Visualization-Machine-Learning-Study-of-DASS-21

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `NumPy`
* **Statistical Analysis:** `SciPy` (Chi-Square Contingency)
* **Machine Learning:** `Scikit-Learn` (Random Forest, Decision Trees, Train-Test Split)
* **Visualization & Diagnostics:** `Matplotlib`, `Seaborn`, `Yellowbrick`

## 📊 Dataset Architecture
The dataset comprises ~40,000 observations from the Depression Anxiety Stress Scale (DASS). Features include:
* **DASS Questions (Q1-Q42):** 4-point Likert scale (Variable A), Response Latency (Variable E), and Question Position (Variable I).
* **TIPI (Big Five):** 10-item personality inventory.
* **Metadata:** 3 validity check variables (VCL), demographic metadata, and server-side timing (elapse).

## ⚙️ Data Engineering & Cleaning Pipeline
1. **Integrity Filtering:** Implemented a verification gate using non-existent control words (`VCL6`, `VCL9`, `VCL12`). Observations with affirmative responses to these "catch" items were purged to mitigate noise from non-serious respondents.
2. **Outlier Mitigation:** Filtered age distributions ($age < 101$) and handled unlisted categorical responses (0) as "No Response" to preserve sample volume without skewing mean values.
3. **Feature Construction:** * Calculated aggregate **Big Five Dimensions** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) by applying reverse-scoring logic to the TIPI scale.
    * **Target Binarization:** In the final iteration, the 4-class target (Q13A) was mapped to a binary set (0: Less Likely, 1: More Likely) to optimize precision-recall for clinical screening utility.

## 🔬 Statistical Methodology
* **Hypothesis Testing:** Conducted **Chi-Squared Tests of Independence** across all Big Five dimensions against Education, Religion, and Orientation.
* **Results:** Rejected $H_0$ in all cases ($p < 0.05$), confirming that personality traits are statistically dependent on demographic backgrounds within this cohort.
* **Correlation Analysis:** Generated a Pearson correlation matrix identifying `Q26` (Down-hearted/blue) and `Q34` (Worthlessness) as the primary features with the highest mutual information relative to the target `Q13A`.

## 🤖 Machine Learning Pipeline
### Model Architecture
I compared a baseline Decision Tree against an optimized Random Forest Classifier.

### Hyperparameter Optimization
Used `Yellowbrick.validation_curve` to analyze the bias-variance tradeoff.
* **Optimization:** `max_depth` was tuned to 9 to prevent overfitting while capturing non-linear relationships.
* **Estimators:** Set to 150 for stable variance reduction.



### Performance Metrics
* **Multi-class Accuracy:** 67% (Weighted F1-score).
* **Binary Accuracy:** 82% (Grade A classification).
* **Class-Specific Analysis:** The model demonstrated high precision in identifying **Class 4 (Severe)** instances, validating its use-case as a high-sensitivity screening tool.
* **Cross-Validation:** Executed 5-fold CV to confirm model stability and ensure results were not a product of data leakage or spatial bias in the training set.
