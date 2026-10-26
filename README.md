# 🏡 Seattle House Price Prediction & Classification (v1, v2 & v3)

An end-to-end Machine Learning journey implementing multiple regression and classification algorithms completely from scratch using pure NumPy (cost functions, analytical gradients, vectorized gradient descent, eigenvalue stability analysis, and L2 regularization) and thoroughly benchmarked against scikit-learn.

---

## 📌 Executive Summary & Architecture

| Stage | Paradigm | Core Objective | Key Features | Model Metric | vs. Baseline |
|---|---|---|---|---|---|
| **v1 Baseline** | Regression | Predict dollar price via `log(price)` | 8 core numeric features, loop-based GD | **Test RMSE: 0.3940**, $R^2$: 0.491 | Baseline RMSE: 0.5522 (+28.7% better) |
| **v2 Advanced** | Regression | Feature engineering & 1000× speedup | 55 features (+age, +renovated, +44 cities) | **Test RMSE: 0.2721**, $R^2$: **0.7217** | Baseline RMSE: 0.5158 (+47.2% better) |
| **v3 Logistic** | Binary Classification | Classify: Expensive ($1$) vs. Affordable ($0$) | 55 features, Sigmoid, Log Loss, $L_2$ Regularization | **Test Accuracy: 85.32%** | Dumb Baseline: 49.34% (**+36% gain**) |

---

## 📊 Stage 1: Linear Regression Baseline (v1)

- **Notebook:** `House_price_prediction_by_linear_reg(v1).ipynb`
- **Data:** 4,549 cleaned listings after dropping 49 zero-price listings and 2 zero-bedroom/bathroom entries.
- **Features (8):** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`.
- **Target:** $\log(\text{price})$ to stabilize the right-skewed pricing distribution.
- **Implementation:** Loop-based Gradient Descent ($\alpha = 0.05, 2000$ iterations).
- **Results:**
  - Cost reduced from $85.47 \to 0.0734$.
  - Test RMSE: **0.3940**, $R^2$: **0.4909**.
  - Verified exact match with Scikit-Learn `LinearRegression` (`np.allclose = True`).
- **Learning Rate Limit:** Theoretical convergence limit $\alpha < 2/\lambda_{\max} \approx 0.759$ experimentally verified ($\alpha = 0.7$ converged, $\alpha = 0.8$ diverged).

---

## 🚀 Stage 2: Advanced Pipeline & Ablation Study (v2)

- **Notebook:** `Advanced_linear_reg(v2).ipynb`
- **Outlier Trimming:** Filtered listings above the 99.5th percentile of `price_per_sqft` ($\approx \$675.13/\text{sqft}$), trimming 23 extreme listings ($4,549 \to 4,526$ rows).
- **Feature Engineering:**
  - Verified exact collinearity: $\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$, dropping $\text{sqft\_above}$.
  - Derived binary $\text{renovated} = (\text{yr\_renovated} > 0)$ and $\text{age} = 2014 - \text{yr\_built}$.
  - One-hot encoded 44 cities, expanding feature space to **55 features**.
- **Vectorized Gradient Descent:** Replaced Python loops with NumPy BLAS matrix operations ($X^T \cdot \text{err}$), achieving a **1000× speedup** (~0.07 ms per step vs. ~60 ms loop-based).
- **Ablation Study Results:**
  | Model | Feature Set | Test RMSE | $R^2$ Score | Key Finding |
  |---|---|---|---|---|
  | **Model A** | 8 original features | 0.3659 | 0.4965 | Trimming outliers alone improved RMSE from 0.3940 $\to$ 0.3659 |
  | **Model B** | 11 features (+basement, renovated, age) | 0.3547 | 0.5270 | Engineered features added clean non-linear signals (+3.1% $R^2$) |
  | **Model C** | 55 features (+44 city one-hot columns) | **0.2721** | **0.7217** | Geographical location provided massive predictive power (+19.5% $R^2$) |
- **Eigenvalue Limit Shift:** With 55 features, the theoretical stability boundary shifted to $2/\lambda_{\max} \approx 0.602$. Confirmed: $\alpha = 0.5$ converged rapidly, while $\alpha = 0.7$ diverged to $9.25 \times 10^9$.
- **Error Analysis:** Diagnosed top-10 worst test predictions; identified that linear regression's additive nature over-predicted large-lot, low-cost listings (e.g., an $\$84\text{k}$ house predicted at $\$1.67\text{M}$).

---

## 🎯 Stage 3: Logistic Regression & L2 Regularization (v3)

- **Notebook:** `House_price_pred_by_logis_reg.ipynb`
- **Problem Formulation:** Partitioned houses by the median price ($\$464,000$) to create a perfectly balanced binary target:
  $$y = \begin{cases} 1 & \text{if price} \ge \text{median (Expensive, 50.06\%)} \\ 0 & \text{if price} < \text{median (Affordable, 49.94\%)} \end{cases}$$
- **From-Scratch Math:**
  - Sigmoid Activation: $g(z) = \frac{1}{1 + e^{-z}}$
  - Binary Cross-Entropy (Log Loss) with numerical stability clipping ($10^{-15} \le f \le 1 - 10^{-15}$):
    $$J(w, b) = -\frac{1}{m} \sum \Big[ y \log(f) + (1 - y) \log(1 - f) \Big] + \frac{\lambda}{2m} \sum w_j^2$$
  - Gradient: $\frac{\partial J}{\partial w} = \frac{1}{m} X^T(f - y) + \frac{\lambda}{m}w$
- **Classification Performance:**
  - **Dumb Majority Baseline Accuracy:** 49.34%
  - **Scratch Logistic Regression Accuracy:** **85.32%** (beating Scikit-Learn's default $85.10\%$)
  - **Confusion Matrix:**
    $$\begin{bmatrix} 392 & 67 \\ 66 & 381 \end{bmatrix} \implies \text{Affordable Recall: } 85.4\%, \text{ Expensive Recall: } 85.2\%$$
- **L2 Regularization Experiments:**
  - $\lambda = 0$: Test Acc = $85.32\%$, $\|w\|_2 = 6.45$
  - $\lambda = 1$: Test Acc = $85.32\%$, $\|w\|_2 = 1.87$ (controlled shrinkage)
  - $\lambda = 100$: Test Acc = $83.77\%$, $\|w\|_2 = 1.25$ (**Underfitting verified**)
  - Tested small-sample regime ($m=50 < n=55$) to demonstrate how regularization restricts weight explosion in data-scarce conditions.

---

## 🛠️ Project Structure

```
House_price_prediction/
├── House_price_prediction_by_linear_reg(v1).ipynb   # v1 Baseline linear regression (8 features, loops)
├── Advanced_linear_reg(v2).ipynb                    # v2 Advanced linear regression (55 features, vectorized)
├── House_price_pred_by_logis_reg.ipynb              # v3 Logistic regression classification (Sigmoid, L2)
├── app.py                                           # Streamlit interactive deployment application
├── training.csv                                     # Kaggle dataset (git-ignored for license compliance)
├── requirements.txt                                 # Python dependencies
└── README.md                                        # Complete engineering documentation
```

---

## 💻 How to Run

### 1. Run the Jupyter Notebooks
```bash
jupyter lab
```
Open and run `House_price_prediction_by_linear_reg(v1).ipynb`, `Advanced_linear_reg(v2).ipynb`, or `House_price_pred_by_logis_reg.ipynb`.

### 2. Launch the Streamlit Interactive Web App
```bash
streamlit run app.py
```



