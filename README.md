# House price prediction with linear regression (v1 & v2)

Predicts the sale price of houses in the Seattle area using multiple linear regression. Implemented completely from scratch with NumPy (cost function, gradients, gradient descent) and rigorously verified against scikit-learn.

- **v1:** Baseline 8-feature scratch linear regression (loop-based).
- **v2:** Full pipeline upgrade with outlier trimming, feature engineering, city one-hot encoding (55 features), 1000× faster vectorized gradient descent, alpha eigenvalue analysis, and error analysis.

---

## Results & Ablation Study

Evaluated on the trimmed dataset (4,526 houses, 80/20 train/test split: 3,620 train / 906 test). The target is modelled as `log(price)`. Baseline test RMSE is **0.5158** (predicting `y_train.mean() = 13.0555`).

| Model | Features | Test RMSE | R² | Key Takeaway |
|---|---|---|---|---|
| **Baseline** (predict train mean) | 0 | 0.5158 | 0.000 | Safe benchmark |
| **Model A** (v1 original 8 features) | 8 | 0.3659 | 0.4965 | Trimming outliers alone drops RMSE from 0.3940 → 0.3659 |
| **Model B** (+ sqft_basement, renovated, age) | 11 | 0.3547 | 0.5270 | Engineered features add genuine signal (+3.1% R²) |
| **Model C / v2 Scratch** (+ 44 city one-hot) | 55 | **0.2721** | **0.7217** | Location provides massive predictive power (+19.5% R²) |
| **scikit-learn LinearRegression (Model C)** | 55 | 0.2721 | 0.7216 | Scratch weights, bias & predictions match (`np.allclose = True`) |

> ⚠️ **Important ML Note on Comparisons:**  
> Never directly compare the raw v1 test RMSE (0.3940 on 910 houses) with v2 Model C (0.2721 on 906 houses) because trimming outliers changes the test set. The fair comparison is **Model A (0.3659)** vs **Model C (0.2721)** on the exact same dataset, showing a **25.6% reduction in RMSE** purely from feature engineering!

---

## What Changed in v2?

### 1. Exploratory Data Analysis (EDA)
- **Log transformation:** Raw prices are heavily right-skewed (skewness 5.3); `log(price)` compresses high values into a near-symmetric bell curve (skewness 0.19).
- **Scatter analysis:** `sqft_living vs price` revealed severe price-per-square-foot outliers.
- **Location signal:** City-wise median prices revealed a 3× difference between cities (e.g. Medina vs Auburn), strongly motivating one-hot encoding.

### 2. Outlier Trimming
- Created `price_per_sqft = price / sqft_living`.
- Filtered out listings above the 99.5th percentile (~$675.13/sqft), dropping 23 extreme listings (4,549 → 4,526 rows).
- This keeps extreme edge cases from dominating the MSE cost function.

### 3. Feature Engineering & Redundancy Removal
- Verified exact collinearity: `sqft_living == sqft_above + sqft_basement` across all rows, allowing `sqft_above` to be dropped without losing information.
- Created `renovated = (yr_renovated > 0).astype(int)` to provide a clean binary signal.
- Created `age = 2014 - yr_built` to capture house age directly.
- Dropped uninformative columns: `date`, `street`, `statezip`, `country`, `yr_built`, `yr_renovated`, `sqft_above`, and `price_per_sqft`.
- One-hot encoded `city` (44 categories), expanding feature space from 11 to **55 features**.

### 4. Vectorization (1000× Speedup)
- Replaced nested Python loops with pure NumPy matrix operations:
  - Prediction: $\vec{f} = Xw + b$
  - Cost: $J = \frac{1}{2m} \sum (f - y)^2$
  - Gradient: $dj\_dw = \frac{1}{m} X^T \cdot (f - y)$ and $dj\_db = \text{mean}(f - y)$
- **Speed benchmark:** Loop version took ~0.06 s per gradient step (~2+ minutes for 2,000 iterations). Vectorized version runs in ~0.07 ms per step (**~1000× faster**, finishing 2,000 iterations in under 1 second).

### 5. Learning Rate ($\alpha$) Limit Shift
Gradient descent converges when $\alpha < 2 / \lambda_{max}$, where $\lambda_{max}$ is the largest eigenvalue of the normalized feature covariance matrix:
- **v1 (8 features):** Limit was $\approx 0.759$ ($\alpha = 0.7$ converged, $\alpha = 0.8$ diverged).
- **v2 (55 features):** Feature geometry changed; limit lowered to $\approx \mathbf{0.602}$.
  - Experiment with $\alpha = 0.5$ ($< 0.602$): cost rapidly drops to 0.0363 in 15 iterations.
  - Experiment with $\alpha = 0.7$ ($> 0.602$): cost explodes from 7.8 → 9.25 Billion in 45 iterations (divergence verified).

### 6. Error Analysis
Inspected the top 10 worst test predictions:
- Largest error was **2.99 log units** on an $84,350 home predicted at **$1,678,479** (~20× over-prediction).
- Top errors were predominantly **over-predictions** on cheap houses. A linear model is purely additive ($w_1 \cdot \text{sqft} + w_2 \cdot \text{city} + \dots$): when a house has high sqft and a good lot in a decent city, linear regression forces a high valuation, unable to capture non-linear penalties (e.g. distressed condition, unusual title/foreclosure sales).

---

## Dataset

- **Source:** Kaggle, [House price prediction](https://www.kaggle.com/datasets/shree1992/housedata) by shree1992. 4,600 rows and 18 columns of house sales in Seattle area, WA (2014).
- **License:** Unknown on Kaggle; data is not committed to this repo. Download and save as `training.csv`.
- **Cleaning:** 49 rows with price 0 and 2 rows with 0 bedrooms/bathrooms removed (4,549 rows). In v2, 23 price/sqft outliers were also removed (4,526 rows).

---

## Project Structure

```
House_price_prediction.ipynb   v1 from-scratch linear regression (8 features, loops)
v2_house.ipynb                 v2 pipeline (55 features, vectorized, ablation, error analysis)
training.csv                   raw dataset (download from Kaggle)
README.md                      full project documentation
```

## How to Run

```bash
jupyter lab
```
Run `House_price_prediction.ipynb` for v1, or `v2_house.ipynb` for v2.


