import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import joblib

# custom scorers to revert log-predictions
def mae_exp(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

def rmse_exp(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))

scoring = {
    'MAE': make_scorer(mae_exp, greater_is_better=False),
    'RMSE': make_scorer(rmse_exp, greater_is_better=False),
    'R2': 'r2'
}

# constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
data_folder_path = "data"
TARGET = "PM2.5"
LOG_TARGET = f'log_{TARGET}'

# ---------------------------
# STEP 1: Load the data
# ---------------------------
def load_preprocessed():
    # data import
    df_train = pd.read_csv(os.path.join(data_folder_path, 'processed', "02_df_train_imputed.csv"), index_col=0)
    df_test = pd.read_csv(os.path.join(data_folder_path, 'processed', "02_df_test_imputed.csv"), index_col=0)

    return df_train, df_test

# ---------------------------
# STEP 3: Model autocorrelation
# ---------------------------
def model_autocorrelation(df_train, df_test):
    # --- Adding lags ---
    cols_to_lag = [LOG_TARGET, 'PM10']
    lag_degree = 1

    for col in cols_to_lag:
        df_train[f"{col}_lag_{lag_degree}"] = df_train[col].shift(lag_degree)
        df_test[f"{col}_lag_{lag_degree}"] = df_test[col].shift(lag_degree)

    df_train = df_train.iloc[lag_degree:]
    df_test = df_test.iloc[lag_degree:]

    return df_train, df_test

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    # ---- STEP 1: Load the data ----
    df_train, df_test = load_preprocessed()
    print("✅Data loaded")

    # ---- STEP 2: Log-transform the target, ensure consistency ----
    df_train[TARGET] = np.log(df_train[TARGET])
    df_test[TARGET] = np.log(df_test[TARGET])

    df_train = df_train.rename(columns={TARGET: LOG_TARGET})
    df_test = df_test.rename(columns={TARGET: LOG_TARGET})
    print("✅Target prepared")

    # ---- STEP 3: Model autocorrelation ----
    df_train, df_test = model_autocorrelation(df_train, df_test)

    X_train = df_train.drop(columns=[LOG_TARGET])
    y_train = df_train[LOG_TARGET]

    X_test = df_test.drop(columns=[LOG_TARGET])
    y_test = df_test[LOG_TARGET]
    print("✅Autocorrelation modelled")

    # ---- STEP 4: Model fit ----
    best_lgbm_regressor_params = {
        'learning_rate': 0.050082407314873414,
        'num_leaves': 37,
        'max_depth': 7,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.9504633949190936,
        'bagging_fraction': 0.9797037164659682,
        'bagging_freq': 3,
        'lambda_l1': 6.160957792389373e-07,
        'lambda_l2': 0.043553728750044186,
        'n_estimators': 680
    }

    print("Fitting the model...")
    best_lgbm_model = lgb.LGBMRegressor(**best_lgbm_regressor_params)
    best_lgbm_model.fit(X_train, y_train)
    print("✅Model trained")

    # ---- STEP 5: Predicting, metric computation and rescaling predictions ----
    best_lgbm_predictions = best_lgbm_model.predict(X_test)

    mae = mae_exp(y_test, best_lgbm_predictions)
    rmse = rmse_exp(y_test, best_lgbm_predictions)
    r2 = r2_score(y_test, best_lgbm_predictions)

    print(f"Test \tMAE: {mae}")
    print(f"Test \tRMSE: {rmse}")
    print(f"Test \tR2: {r2}")

    df_test[f'{TARGET}_pred'] = np.exp(best_lgbm_predictions)
    print("✅Predicted and reverted")

    # ---- STEP 6: Saving the model and predictions ----
    # saving numerical imputation model (to heavy to save)
    # joblib.dump(best_lgbm_model, os.path.join(data_folder_path, 'models', "PM2.5_best_lgbm_model.pkl"))

    df_test.to_csv(os.path.join(data_folder_path, 'processed', "04_df_test_predicted.csv"), index=True, index_label='datetime')
    print("✅ Prediction finished. Predicted .csv saved to 'data/processed/'04_df_test_predicted.")


if __name__ == "__main__":
    print("Processing...")
    main()