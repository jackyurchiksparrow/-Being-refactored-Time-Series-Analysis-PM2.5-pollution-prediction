import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer

def mae_exp(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

def rmse_exp(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))

scoring = {
    'MAE': make_scorer(mae_exp, greater_is_better=False),
    'RMSE': make_scorer(rmse_exp, greater_is_better=False),
    'R2': 'r2'
}

data_folder_path = "data\\DataHourlyChina"
TARGET = "PM2.5"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df = pd.read_csv(os.path.join(data_folder_path, "POST_EDA_POST_FEAT_ENG_STANDARDIZED_TARGET_TRANSFORMED.csv"), index_col=0)


X_train, X_test, y_train, y_test = train_test_split(df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=RANDOM_SEED)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

weights = np.where(y_train > y_train.quantile(0.7), 4.5, 1)  # Give 4.5x weight to large targets

best_cat_model = CatBoostRegressor(
    n_estimators=1500,
    max_depth=13,
    learning_rate=0.16,
    random_seed=RANDOM_SEED,
    verbose=0
)

print("Training will take about 5 minutes. I'm on it...")
best_cat_model.fit(X_train, y_train, sample_weight=weights)
predictions = best_cat_model.predict(X_test)

mae = mae_exp(y_test, predictions)
rmse = rmse_exp(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test \tMAE: {mae}")
print(f"Test \tRMSE: {rmse}")
print(f"Test \tR2: {r2}")

best_cat_model.save_model('best_cat_model.cbm')
print("Done. The model saved")