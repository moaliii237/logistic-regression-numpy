# Logistic Regression from Scratch (NumPy)

This is my Year 1, Block B machine learning project (FAI1, 2024-25). The main thing I built here is logistic regression from scratch in NumPy, so parameter init, the sigmoid, the binary cross-entropy cost, the gradient descent update, prediction with a threshold, and an accuracy function. I wrote both the plain version and an L2 regularized version, trained them on a real hospital dataset, and then checked my numbers against scikit-learn's `LogisticRegression` on the same split to make sure my math was actually correct.

I did this project on my own.

The point was not to beat scikit-learn. The point was to understand what is happening inside it, so I implemented every step myself with the vectorized formulas instead of calling `.fit()` and trusting it.

## What it does

- Loads and cleans the Diabetes 130-US Hospitals dataset (about 101,766 rows after I drop the columns I do not use and drop missing rows).
- Picks a binary target and four features, splits into train/test, and scales the features.
- Implements logistic regression in NumPy from scratch:
  - `Init_param(p)` to make the weight vector with `p + 1` values (the extra one is the bias).
  - `sig_fun(X, W)` the sigmoid, vectorized as `1 / (1 + e**(-X @ W))`.
  - `cost_fun(X, Y, W)` binary cross-entropy in the vectorized form, with `np.clip` to stop `log(0)`.
  - `grad_descent_step_wr(...)` one gradient descent step using `W - (alf/m) * X.T @ (Y_pred - Y)`.
  - a training loop that runs the step N times and saves the cost every iteration so I can plot convergence.
  - `pred_fun(X, W, threshold=0.5)` turns probabilities into 0/1 labels.
  - `eval_fun(...)` computes accuracy as `np.mean(Y == Y_pred)`.
- Adds the L2 regularized version: `cost_fun_reg` and `grad_descent_step_reg`, which add the penalty term but skip the bias using a `PI` matrix built with `np.block`, `np.zeros`, and `np.eye` so the intercept is not penalized.
- Compares my from-scratch weights and accuracy against scikit-learn `LogisticRegression` trained on the exact same data.
- Does basic hyperparameter tuning by hand (learning rate, number of iterations, lambda) and records the sets I tried.

There is also a second, larger notebook (the creative brief) where I go through the full block: EDA with SQL queries and with Python, data cleaning, then regression, classification, and clustering baselines, plus feature selection and tuning. More on that below.

## The dataset

I used the **Diabetes 130-US Hospitals (1999-2008)** dataset from the UCI Machine Learning Repository.

- UCI page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- File the notebooks expect: `diabetic_data.csv`

The dataset is not in this repo. Download `diabetic_data.csv` from UCI and put it in the `data/` folder so the path `./data/diabetic_data.csv` works. The `math/` notebooks also use a small `house-prices.csv`, put that in `data/` too as `./data/house-prices.csv` if you want to run it.

## How it works (my approach)

For the cleaning step I dropped a lot of columns that were mostly empty or that I was not going to use (the drug columns like `troglitazone` and `examide`, the `weight`, `payer_code`, `medical_specialty`, the diagnosis codes, and the ID columns), then `dropna()`. I remapped some categorical columns to numbers by hand, for example `insulin` to `{No:0, Steady:1, Down:2, Up:3}` and the discharge and admission source IDs into smaller groups, because the raw IDs had too many tiny categories.

For the model part I chose `diabetesMed` (does the patient take diabetes medication) as the target, and `insulin`, `num_medications`, `change`, and `time_in_hospital` as the four features. I split with `train_test_split(test_size=0.2, random_state=58)`, which gives 81,412 training rows and 20,354 test rows. I scaled the features with `MinMaxScaler` into the 0 to 1 range, then added a column of ones to the front of `X` with `np.insert` so the bias term `w0` is just part of the matrix multiply. I set `np.random.seed(1979)` so the random weight init is reproducible.

A few things I had to be careful about:

- **Shapes.** This bit me a lot. The template wants `X` and `Y` as 2D arrays, the weights as 1D, and the gradient step needs `W` reshaped with `np.expand_dims` to line up with `X.T @ (Y_pred - Y)`, then squeezed back. Getting these consistent was most of the debugging.
- **`log(0)`.** The cost blows up if a prediction is exactly 0 or 1, so I clip predictions to `[1e-15, 1 - 1e-15]` before taking the log.
- **Not penalizing the bias.** In the regularized version the penalty should only hit the real feature weights, not `w0`. I build the `PI` matrix so the first diagonal entry is 0, which leaves the intercept out of the L2 term.
- **Why `alpha = 1` and `N = 70000`.** With smaller learning rates and fewer iterations the cost was still going down and had not flattened, so accuracy was stuck around 0.77 (see the tuning table in the notebook). Pushing the learning rate to 1 and running 70,000 iterations let it actually converge to about 0.85. I could see this directly on the cost-vs-iteration plot, that is why I keep that plot in the notebook.

I tested each function against the expected values the template gives (for example `cost_fun` should return `2.27638632` on the test input, and my code returns exactly that), so I knew each piece was right before plugging it into the training loop.

## Tech stack

- NumPy (the from-scratch model, all the math)
- pandas (loading and cleaning)
- scikit-learn (`train_test_split`, `MinMaxScaler`, and `LogisticRegression` only as a reference to check my work)
- matplotlib and seaborn (cost plots, coefficient plots, EDA charts)
- XGBoost (used in the creative brief notebook for the regression baselines)
- joblib (saving the best model in the creative brief)

## Results

All on the held-out test set (20,354 rows):

- My from-scratch logistic regression (no regularization): accuracy **0.8514**
- My from-scratch logistic regression with L2 (lambda 30): accuracy **0.8516**
- scikit-learn `LogisticRegression` on the same split: accuracy **0.8514**

So my implementation matches scikit-learn on accuracy, which was the goal. The weight vectors are close but not identical. Mine came out around `[-0.79, 22.6, 2.98, 9.12, -0.55]` and scikit-learn's around `[-0.78, 17.6, 2.91, 7.22, -0.52]`. The signs and rough magnitudes agree. The difference is mostly because scikit-learn applies its own regularization by default and uses a different solver, while my plain version uses none, so my coefficients are allowed to grow larger. Adding the L2 penalty pulls my coefficients down (to about `[-0.54, 9.06, 1.81, 4.34, -0.28]`) without hurting accuracy, which is what I expected.

Honest limitations:

- The 0.85 accuracy is partly because the target `diabetesMed` is not balanced, so a model can do well just by leaning toward the majority class. I report accuracy because that is what the assignment asked for, but for this kind of target a metric like F1 would tell a more honest story.
- I only used four features, so this is a deliberately simple model, not a tuned production one.
- My hand tuning was a small grid (three sets), not an exhaustive search.

## How to run

```
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook you want. Make sure `data/diabetic_data.csv` exists first (see the dataset section), otherwise the first cell fails.

## Main notebooks

- `logistic_regression_numpy.ipynb` - the from-scratch logistic regression. This is the one to read first.
- `creative_brief_full_block.ipynb` - the full block notebook. It has EDA in SQL and in Python, data cleaning and preprocessing, regression baselines (Linear Regression, Decision Tree, Gradient Boosting, XGBoost), classification baselines (Logistic Regression, Random Forest, KNN), clustering (KMeans, Agglomerative, DBSCAN with PCA), feature engineering and selection, and hyperparameter tuning with `GridSearchCV` plus nested cross-validation, then saving the best model with joblib.

## Note on the extra files

