import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- Constants ---
data_folder_path = "data"
TARGET = "PM2.5"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Utility imports ---
from utils import (
    get_a_mask_for_features, test_imputer_scores, impute_columns, CategoricalImputer
)

# ---------------------------
# STEP 1: Load and clean
# ---------------------------
def load_and_clean():
    df = pd.read_csv(os.path.join(data_folder_path, 'raw', "PRSA_Data_Aotizhongxin_20130301-20170228.csv"), index_col=0)
    df.drop(columns=["station"], inplace=True)

    return df

# ---------------------------
# STEP 2: Train-test split
# ---------------------------
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[TARGET],
        test_size=0.2,
        shuffle=False,
        random_state=RANDOM_SEED
    )
    df_train, df_test = X_train.copy(), X_test.copy()
    df_train[TARGET] = y_train
    df_test[TARGET] = y_test

    # set datetime index
    df_train.index = pd.to_datetime(df_train[['year', 'month', 'day', 'hour']])
    df_test.index = pd.to_datetime(df_test[['year', 'month', 'day', 'hour']])

    return df_train, df_test


# ---------------------------
# STEP 3: Feature engineering
# ---------------------------
def feature_engineering(df_train, df_test):

    def encode_season(month):
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        elif month in [9, 10, 11]:
            return 4  # Autumn
    
    # Season, weekend, working time
    df_train['season'] = df_train.index.month.map(encode_season)
    df_test['season'] = df_test.index.month.map(encode_season)

    df_train['is_weekend'] = df_train.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)
    df_test['is_weekend'] = df_test.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)

    df_train['dayofweek'] = df_train.index.dayofweek
    df_test['dayofweek'] = df_test.index.dayofweek

    df_train['is_working_time'] = df_train['hour'].apply(lambda x: 1 if x >= 6 and x <= 18 else 0)
    df_test['is_working_time']  = df_test['hour'].apply(lambda x: 1 if x >= 6 and x <= 18 else 0)

    # Binning wind direction
    bin_map = {
        'N': 'N', 'NNE': 'N', 'NNW': 'N',
        'NE': 'NE', 'ENE': 'NE',
        'E': 'E', 'ESE': 'E',
        'SE': 'SE', 'SSE': 'SE',
        'S': 'S', 'SSW': 'S',
        'SW': 'SW', 'WSW': 'SW',
        'W': 'W', 'WNW': 'W',
        'NW': 'NW'
    }
    df_train['wd_binned'] = df_train['wd'].map(bin_map)
    df_test['wd_binned'] = df_test['wd'].map(bin_map)

    df_train.drop(columns=['wd'], inplace=True)
    df_test.drop(columns=['wd'], inplace=True)

    return df_train, df_test


# ---------------------------
# STEP 4: Outlier handling
# ---------------------------
def winsorize_outliers(df_train, df_test):
    to_winsorize = {
        'PM10': (0.00001, 0.001), 'SO2': (0.00001, 0.001),
        'NO2': (0.00001, 0.001), 'CO': (0.00001, 0.001),
        'O3': (0.00001, 0.001), 'PM2.5': (0.00001, 0.0001),
        'TEMP': (0.001, 0.0005), 'PRES': (0.0001, 0.001),
        'DEWP': (0.001, 0.0008), 'RAIN': (0.00001, 0.001),
        'WSPM': (0.00001, 0.0001),
    }
    winsor_limits = {}
    for feat, (low_q, high_q) in to_winsorize.items():
        X = df_train[feat]
        lower = X.quantile(low_q)
        upper = X.quantile(1 - high_q)
        winsor_limits[feat] = (lower, upper)

    for feat, (lower, upper) in winsor_limits.items():
        df_train[feat] = df_train[feat].clip(lower, upper)
        df_test[feat] = df_test[feat].clip(lower, upper)

    return df_train, df_test

# ---------------------------
# STEP 4: Cyclical feature encoding
# ---------------------------
def encode_cyclical_features(dataframe: pd.DataFrame):
    df_index = pd.to_datetime(dataframe.index)

    dataframe['hour_sin'] = np.sin(2 * np.pi * df_index.hour / 24)
    dataframe['hour_cos'] = np.cos(2 * np.pi * df_index.hour / 24)

    dataframe['month_sin'] = np.sin(2 * np.pi * df_index.month / 12)
    dataframe['month_cos'] = np.cos(2 * np.pi * df_index.month / 12)

    day_of_month = df_index.day                    # 1..28/29/30/31
    days_in_month = df_index.days_in_month        # 28..31
    # avoid division by zero if single-day month (theoretically impossible), use days_in_month - 1 for range
    day_fraction = (day_of_month - 1) / (days_in_month - 1)   # 0.0 .. 1.0

    dataframe["day_sin"] = np.sin(2 * np.pi * day_fraction)
    dataframe["day_cos"] = np.cos(2 * np.pi * day_fraction)

    dataframe['dayofweek_sin'] = np.sin(2 * np.pi * dataframe['dayofweek'] / 7)
    dataframe['dayofweek_cos'] = np.cos(2 * np.pi * dataframe['dayofweek'] / 7)

    # Due to its circular nature, it's closer to nominal than ordinal. It is directional, we are representing it in degrees (0-360):
    direction_map = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180,
        'SW': 225, 'W': 270, 'NW': 315
    }

    dataframe['wd_deg'] = dataframe['wd_binned'].map(direction_map)

    dataframe.drop(columns=["hour", "day", "month", "dayofweek"], inplace=True)


# ---------------------------
# STEP 7: Data preprocessing
# ---------------------------
def preprocess(df_tr, df_te, target: str,
               numerical_cols, categoric_nominal_cols, categoric_ordinal_cols,
               to_scale=True, drop_first=True, fill_na=True):
    
    scaler_df = None
    # convert arrays to numpy
    X_numerical_cols = np.array(numerical_cols)
    X_categoric_nominal_cols = np.array(categoric_nominal_cols)
    X_categoric_ordinal_cols = np.array(categoric_ordinal_cols)

    ytr_prepared = df_tr[target]
    yte_prepared = df_te[target]
    
    # ensure target not in feature arrays
    X_numerical_cols = X_numerical_cols[X_numerical_cols != target]
    X_categoric_nominal_cols = X_categoric_nominal_cols[X_categoric_nominal_cols != target]
    X_categoric_ordinal_cols = X_categoric_ordinal_cols[X_categoric_ordinal_cols != target]

    if target not in df_tr.columns:
        print(f"{target} wasn't found in train df, please add it")
        return
    if target not in df_te.columns:
        print(f"{target} wasn't found in test df, please add it")
        return

    # --- NUMERIC
    Xtr_num = df_tr[X_numerical_cols]
    Xte_num = df_te[X_numerical_cols]
    if fill_na:
        train_medians = Xtr_num.median()
        Xtr_num = Xtr_num.fillna(train_medians)
        Xte_num = Xte_num.fillna(train_medians)

    if to_scale:
        scaler = StandardScaler().fit(Xtr_num)
        scaler_df = pd.DataFrame({'col': numerical_cols, 'std': scaler.scale_})
        Xtr_num = pd.DataFrame(scaler.transform(Xtr_num), columns=numerical_cols, index=Xtr_num.index)
        Xte_num = pd.DataFrame(scaler.transform(Xte_num), columns=numerical_cols, index=Xte_num.index)

    # --- CATEGORICAL
    X_cat_columns = np.append(X_categoric_nominal_cols, X_categoric_ordinal_cols)
    Xtr_cat = df_tr[X_cat_columns].copy()
    Xte_cat = df_te[X_cat_columns].copy()
    if fill_na:
        train_modes = Xtr_cat.mode().iloc[0]
        Xtr_cat = Xtr_cat.fillna(train_modes)
        Xte_cat = Xte_cat.fillna(train_modes)

    Xtr_cat_dummies = pd.get_dummies(Xtr_cat[X_categoric_nominal_cols], drop_first=drop_first, dtype=float)
    Xte_cat_dummies = pd.get_dummies(Xte_cat[X_categoric_nominal_cols], drop_first=drop_first, dtype=float)
    Xte_cat_dummies = Xte_cat_dummies.reindex(columns=Xtr_cat_dummies.columns, fill_value=0.0)

    # --- combine final
    Xtr_prepared = pd.concat([Xtr_num, Xtr_cat_dummies, Xtr_cat[X_categoric_ordinal_cols]], axis=1)
    Xte_prepared = pd.concat([Xte_num, Xte_cat_dummies, Xte_cat[X_categoric_ordinal_cols]], axis=1)
    Xte_prepared = Xte_prepared.reindex(columns=Xtr_prepared.columns, fill_value=0.0)

    return Xtr_prepared, Xte_prepared, ytr_prepared, yte_prepared, scaler_df

# ---------------------------
# STEP 8: Feature engineering2
# ---------------------------
def feature_engineering2(df_train, df_test):
    # account for near-perfect multicollinearity in `wd` (it is represented twice; we retain numerical)
    df_train = df_train.drop(columns=['wd_binned_N', 'wd_binned_NE', 'wd_binned_NW', 'wd_binned_S', 'wd_binned_SE', 
                                    'wd_binned_SW', 'wd_binned_W'])
    df_test = df_test.drop(columns=['wd_binned_N', 'wd_binned_NE', 'wd_binned_NW', 'wd_binned_S', 'wd_binned_SE', 
                                    'wd_binned_SW', 'wd_binned_W'])
    
    def relative_humidity(temp_c, dewpoint_c):
        # Magnus formula
        a, b = 17.27, 237.7
        alpha = (a * dewpoint_c) / (b + dewpoint_c)
        beta = (a * temp_c) / (b + temp_c)
        rh = 100 * (np.exp(alpha) / np.exp(beta))
        return rh

    df_train['RH'] = relative_humidity(df_train['TEMP'], df_train['DEWP'])
    df_test['RH'] = relative_humidity(df_test['TEMP'], df_test['DEWP'])
    df_train = df_train.drop(columns=['TEMP', 'DEWP'], axis=1)
    df_test = df_test.drop(columns=['TEMP', 'DEWP'], axis=1)


    df_train['working_hour_component'] = df_train['is_working_time'] * df_train['hour_cos']
    df_test['working_hour_component'] = df_test['is_working_time'] * df_test['hour_cos']
    df_train = df_train.drop(columns=["is_working_time", "hour_cos"])
    df_test = df_test.drop(columns=["is_working_time", "hour_cos"])

    return df_train, df_test

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    # --- Step 1: Load raw data ---
    df = load_and_clean()

    # --- Step 2: Train/test split ---
    df_train, df_test = split_data(df)

    # --- Step 3: Feature engineering (season, weekend, wd_binned, etc.) ---
    df_train, df_test = feature_engineering(df_train, df_test)
    
    # --- Step 4: Outlier winsorization ---
    df_train, df_test = winsorize_outliers(df_train, df_test)

    # --- Step 5: Cyclical encoding ---
    encode_cyclical_features(df_train)
    encode_cyclical_features(df_test)

    # --- Step 6: Imputation (numeric + categorical) ---
    ordinal_categorical_cols = ['is_working_time', 'season', 'is_weekend']
    nominal_categorical_cols = ['wd_binned']
    numerical_cols = df_train\
        .drop(columns=ordinal_categorical_cols + nominal_categorical_cols + [TARGET])\
        .select_dtypes(include="number").columns.tolist()

    # --- Step 7: Data preprocessing ---
    X_df_train, X_df_test, \
    y_df_train, y_df_test, \
    scaler_df = preprocess(df_train, df_test, TARGET, 
                        numerical_cols, nominal_categorical_cols, ordinal_categorical_cols,
                        to_scale=False, drop_first=True, fill_na=False)

    df_train = pd.concat([X_df_train, y_df_train], axis=1)
    df_test  = pd.concat([X_df_test, y_df_test], axis=1)

    # --- Step 8: Feature engineering (working_hour_component, RH, multicollinear dropping) ---
    df_train, df_test = feature_engineering2(df_train, df_test)

    # --- Step 9: Training the best regression imputer and imputation of numericals ---
    numerical_nan_cols = ['CO', 'O3', 'NO2', TARGET, 'SO2', 'PM10', 'PRES', 'RAIN', 'WSPM', 'RH']
    ordinal_nan_cols = ['wd_deg']
    nominal_nan_groups = {}

    jobs = os.cpu_count() - 1

    rf_estimator=RandomForestRegressor(n_estimators=120, 
            max_depth=30, 
            n_jobs=jobs, 
            bootstrap=True, 
            random_state=RANDOM_SEED
    )

    rf_num_imputer = IterativeImputer(
        estimator=rf_estimator,
        max_iter=10,
        initial_strategy='median',
        random_state=RANDOM_SEED
    )

    mask = get_a_mask_for_features(data_frame=df_train,
                                    cont_num_features_to_mask=numerical_nan_cols,
                                    ord_categorical_features_to_mask=[],
                                    nominal_groups_to_mask=[],
                                    rand_seed=RANDOM_SEED,
                                    missing_fraction=0.05)

    print("Training regression imputer...")
    scores_df = test_imputer_scores(
        imputer=rf_num_imputer,
        data_frame=df_train,
        numericals_and_ordinals_to_impute=numerical_nan_cols, # only numericals to impute (regression)
        all_dataframe_nominal_groups=nominal_nan_groups, # specify nominals to correctly evaluate
        mask=mask,
        mode='regression',
        verbose=True
    )

    print("Imputing numericals...")
    df_train_imputed_numericals = impute_columns(
        fitted_imputer=rf_num_imputer,
        dataframe=df_train,
        numericals_and_ordinals_to_impute=numerical_nan_cols,
        nominal_groups_to_impute={},
    )

    df_test_imputed_numericals = impute_columns(
        fitted_imputer=rf_num_imputer,
        dataframe=df_test,
        numericals_and_ordinals_to_impute=numerical_nan_cols,
        nominal_groups_to_impute={},
    )

    # --- Step 10: Training the best classification imputer and imputation of categoricals ---
    rf_cat_imputer = CategoricalImputer(
        ordinal_cols_to_impute=ordinal_nan_cols,
        nominal_cols_to_impute=nominal_nan_groups,
        estimator=RandomForestClassifier(
            n_estimators=1000, 
            random_state=RANDOM_SEED, 
            n_jobs=-1
        ),
        random_state=RANDOM_SEED
    )

    mask = get_a_mask_for_features(data_frame=df_train_imputed_numericals,
                                    cont_num_features_to_mask=[],
                                    ord_categorical_features_to_mask=ordinal_nan_cols,
                                    nominal_groups_to_mask=nominal_nan_groups,
                                    rand_seed=RANDOM_SEED,
                                    missing_fraction=0.05)

    print("Training classification imputer...")
    df_sc_cls = test_imputer_scores(
        imputer=rf_cat_imputer,
        data_frame=df_train_imputed_numericals,
        numericals_and_ordinals_to_impute=ordinal_nan_cols,
        all_dataframe_nominal_groups=nominal_nan_groups,
        mask=mask,
        mode='classification',
        verbose=True
    )

    print("Imputing categoricals...")
    df_train_imputed = impute_columns(
        fitted_imputer=rf_cat_imputer,
        dataframe=df_train_imputed_numericals,
        numericals_and_ordinals_to_impute=ordinal_nan_cols,
        nominal_groups_to_impute=nominal_nan_groups,
    )

    df_test_imputed = impute_columns(
        fitted_imputer=rf_cat_imputer,
        dataframe=df_test_imputed_numericals,
        numericals_and_ordinals_to_impute=ordinal_nan_cols,
        nominal_groups_to_impute=nominal_nan_groups,
    )

    # --- Step 11: Saving the data ---
    #print("Saving the models...")
    # saving numerical imputation model (too heavy to save)
    #joblib.dump(rf_num_imputer, os.path.join(data_folder_path, 'models', "numerical_imputer_random_forest.pkl"))

    # saving categorical imputation model (too heavy to save)
    #joblib.dump(rf_cat_imputer, os.path.join(data_folder_path, 'models', "categorical_imputer_random_forest.pkl"))
    
    print("Saving the processed '.csv's")
    df_train_imputed.to_csv(os.path.join(data_folder_path, 'processed', "02_df_train_imputed.csv"), index=True, index_label='datetime')
    df_test_imputed.to_csv(os.path.join(data_folder_path, 'processed', "02_df_test_imputed.csv"), index=True, index_label='datetime')

    print("✅ Preprocessing finished. Train/test sets saved to 'data/processed/'.")

if __name__ == "__main__":
    print("Processing...")
    main()