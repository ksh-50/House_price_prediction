# House price prediction with linear regression

Predicts the sale price of houses in the Seattle area from eight numeric features. Linear regression is implemented from scratch with NumPy (cost function, gradient, gradient descent) and checked against scikit-learn.

## Results

Evaluated on a test set of 910 houses. Price is modelled as log(price).

| Model | RMSE (log price) | R² |
|---|---|---|
| Baseline: predict the training mean | 0.5522 | n/a |
| Linear regression from scratch | 0.3940 | 0.491 |
| Linear regression with scikit-learn | 0.3940 | 0.491 |

- RMSE is 28.66% lower than the baseline. In price terms, the typical error is about a factor of 1.5 (e^0.394 ≈ 1.48).
- Training RMSE is 0.383, close to the test RMSE of 0.394, so there is little overfitting.
- Weights, bias and predictions from the scratch model match scikit-learn (`np.allclose` returns `True`).
- R² is computed as 1 − MSE(model) / MSE(baseline), with the training mean as the baseline. scikit-learn's `r2_score` uses the test mean instead and gives 0.4909.

## Dataset

- **Source:** Kaggle, [House price prediction](https://www.kaggle.com/datasets/shree1992/housedata) by shree1992. 4,600 rows and 18 columns of house sales in the Seattle area, Washington, in 2014.
- **License:** not stated on Kaggle ("Unknown"), so the data file is not included in this repository. Download it and save it as `training.csv` in the project folder.
- **Cleaning:** removed 49 rows with price 0 and 2 rows with 0 bedrooms or 0 bathrooms, leaving 4,549 rows. Zeros in `waterfront`, `view`, `sqft_basement` and `yr_renovated` mean "none" or "never", so they are kept.
- **Features (8):** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`.
- **Target:** log(price). Raw prices are heavily skewed (skewness 25.0); after the log it is 0.33.

## Approach

1. Clean the data and split it 80/20 (`random_state=42`): 3,639 training rows and 910 test rows.
2. Standardize the features with z-scores. Mean and standard deviation come from the training set only and are applied to both sets.
3. Implement the cost J(w, b) = 1/(2m) · Σ (w·x + b − y)² and its gradient with loops in NumPy.
4. Run gradient descent with α = 0.05 for 2,000 iterations. The cost falls from 85.47 to 0.0734 and is flat after about 200 iterations.
5. Compare against a baseline that predicts the training mean, then against scikit-learn's `LinearRegression`.

## Learning rate experiment

Gradient descent for 100 iterations with six learning rates:

| α | Behaviour |
|---|---|
| 0.01 | Converges slowly: cost is still about 11.5 after 100 iterations |
| 0.05, 0.1 | Converge: cost is about 0.077 and 0.0735 after 100 iterations |
| 0.5 | Fastest: cost is about 0.074 within 10 iterations |
| 0.7 | Converges, with slight oscillation early on |
| 0.8 | Diverges: cost grows from 0.5 to about 5 × 10⁷ |

Gradient descent converges when α < 2/λmax, where λmax is the largest eigenvalue of XᵀX/m. For this data λmax = 2.636, so the limit is about 0.759. That matches the experiment: 0.7 converges and 0.8 does not.

## Example prediction

The model outputs log(price), so `exp` converts it back to dollars. A house with 3 bedrooms, 2 bathrooms, 1,800 sqft of living space, a 5,000 sqft lot, 1 floor, no waterfront, view 0 and condition 3 is predicted at about **$377,215**.

## Notes

- The cost is J = MSE / 2, so RMSE = √(2J).
- Scaling statistics must come from the training set only. Test-set statistics would leak information and put test data on a different scale from the one the model learned.
- A log target keeps a few very expensive houses from dominating the squared error.
- `sqft_living` has the largest weight: 0.322 per standard deviation, about +38% in price with the other features fixed. `bedrooms` has a small negative weight (−0.042), likely because it is correlated with `sqft_living`.

## Limitations and next steps

- Only numeric features are used; location (city, zip code) is not.
- Extreme outliers remain in the data, for example a $26.59M listing for a 1,180 sqft house.
- A single random split is used, with no cross-validation.
- Planned for v2: add city (one-hot) and renovation/age features, handle outliers, vectorize the gradient computation, and compare feature sets in an ablation.

## Run it

```
pip install -r requirements.txt
jupyter lab
```

Open `House_price_prediction.ipynb` and run all cells. `training.csv` must be in the same folder.

```
House_price_v1.ipynb   full analysis notebook
requirements.txt
README.md
training.csv           not included, download from Kaggle
```

