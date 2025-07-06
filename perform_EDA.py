import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer # imperative to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

data_folder_path = "data\\DataHourlyChina"
TARGET = "PM2.5"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


df = pd.read_csv(os.path.join(data_folder_path, "PRSA_Data_Aotizhongxin_20130301-20170228.csv"), index_col=0)
df = df.drop(columns=["station"])

object_features = [feat for feat in df.columns if df[feat].dtype == "O"]
numeric_features = [feat for feat in df.columns if df[feat].dtype != "O"]

outlier_indexes = set()
outlier_indexes.add(df[df['PM10'] == max(df['PM10'])].index[0])
outlier_indexes.add(df[df['PM10'] == max(df['PM10'])].index[0])
outlier_indexes.add(df[df['SO2'] > 200].index[0])
outlier_indexes.add(df[(df['SO2'] == 130) & (df['PM2.5'] == 3)].index[0])
outlier_indexes.add(df[(df['TEMP'] > 20) & (df['PM2.5'] > 600)].index[0])
outlier_indexes.add(df[(df['RAIN'] > 10) & (df['PM2.5'] > 400)].index[0])
outlier_indexes.add(df[df['WSPM'] > 10].index[0])
df.drop(outlier_indexes, inplace=True)


wd_probabilities = df['wd'].value_counts(normalize=True)
wd_nulls = df['wd'].isnull()

df.loc[wd_nulls, 'wd'] = np.random.choice(
    wd_probabilities.index,
    size=wd_nulls.sum(),
    p=wd_probabilities.values
)

imputer_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer_median = SimpleImputer(strategy='median', missing_values=np.nan)

df['PM2.5'] = imputer_mean.fit_transform(df[['PM2.5']])
df['PM10'] = imputer_median.fit_transform(df[['PM10']])
df['NO2'] = imputer_median.fit_transform(df[['NO2']])
df['CO'] = imputer_mean.fit_transform(df[['CO']])
df['PRES'] = imputer_mean.fit_transform(df[['PRES']])

def impute_nulls_based_on_cols(dataframe: pd.DataFrame, imputer_obj: IterativeImputer, column_to_impute: str, correlated_columns: list):
    subset = dataframe[[column_to_impute] + correlated_columns]
    null_mask = subset[column_to_impute].isna()
    imputed_subset = imputer_obj.fit_transform(subset)
    dataframe.loc[null_mask, column_to_impute] = imputed_subset[null_mask, 0]

imputer = IterativeImputer(random_state=42)
impute_nulls_based_on_cols(df, imputer, 'O3', ['PRES', 'TEMP', 'NO2'])
impute_nulls_based_on_cols(df, imputer, 'SO2', ['CO', 'NO2', 'PM2.5', 'PM10'])
impute_nulls_based_on_cols(df, imputer, 'TEMP', ['DEWP', 'PRES', 'O3'])
impute_nulls_based_on_cols(df, imputer, 'DEWP', ['PRES', 'TEMP'])
impute_nulls_based_on_cols(df, imputer, 'WSPM', ['NO2', 'O3'])

df['RAIN'] = imputer_median.fit_transform(df[['RAIN']])

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cols = encoder.fit_transform(df.dropna()[['wd']])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['wd']), index=df.dropna().index)
df = pd.concat([df.drop('wd', axis=1), encoded_df], axis=1)

df = df.drop(columns=['DEWP', 'PRES'])

datetime_index = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.index = datetime_index

df.index = pd.to_datetime(df.index)

# feature engineering
df['dayofweek'] = df.index.dayofweek

df.drop(columns=['month', 'day', 'hour'], inplace=True)

df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)

df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
df = df.drop(columns=['dayofweek'])

df["TEMP_x_CO"] = df["TEMP"] * df["CO"]
df["NO2_x_RAIN"] = df["NO2"] * df["RAIN"]
df["WSPM_X_SO2"] = df["WSPM"] * df['SO2']

df["PM10_diff1"] = df["PM10"].diff(1)
df["TEMP_diff1"] = df["TEMP"].diff(1)
df = df.dropna()

# standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop(columns=[TARGET]))
scaled_df = pd.DataFrame(scaled, columns=df.drop(columns=[TARGET]).columns, index=df.index)
df = pd.concat([scaled_df, df[TARGET]], axis=1)

# log-transform the target
df[TARGET] = np.log(df[TARGET])

# drop the redundant features (based on permutation importance)
df = df.drop(columns=['RAIN', 'wd_ENE', 'wd_ESE', 'wd_N', 'wd_NE', 'wd_NNE', 'wd_NNW',
       'wd_NW', 'wd_S', 'wd_SE', 'wd_SSE', 'wd_SSW', 'wd_SW', 'wd_W', 'wd_WNW',
       'wd_WSW', 'is_weekend', 'NO2_x_RAIN'])

df.to_csv(os.path.join(data_folder_path, "POST_EDA_POST_FEAT_ENG_STANDARDIZED_TARGET_TRANSFORMED.csv"), index=True, index_label='datetime')
print("Done")