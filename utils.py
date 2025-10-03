import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Literal
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_arch
from sklearn.experimental import enable_iterative_imputer # imperative to enable IterativeImputer
from sklearn.impute import IterativeImputer
from numpy.linalg import matrix_rank
from itertools import combinations
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.metrics           import roc_auc_score
from sklearn.linear_model import LinearRegression 
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.genmod.cov_struct import Independence
from statsmodels.tsa.stattools import acf
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score
import math
from tqdm import tqdm
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier


# Checks whether a time series dataset has consistent hourly timestamps and identifies any irregular gaps.
def date_consistency(dataframe: pd.DataFrame, year_col: pd.Series, month_col: pd.Series, day_col: pd.Series, hour_col: pd.Series):
    """
    - Combines year, month, day, and hour columns into a proper datetime.
    - Sorts the dataframe by datetime.
    - Calculates time differences between consecutive records.
    - Prints frequency of observed time intervals (e.g., 1h, 2h, etc.).
    - Plots time gaps over the timeline.

    Parameters:
        dataframe (pd.DataFrame): Input dataset containing time components.
        year_col (pd.Series): Column with year values.
        month_col (pd.Series): Column with month values.
        day_col (pd.Series): Column with day values.
        hour_col (pd.Series): Column with hour values.

    Returns:
        None (prints summary and plots time gaps).
    """
    df_working = dataframe.copy()

    date_dict = {
        'year': year_col,
        'month': month_col,
        'day': day_col,
        'hour': hour_col
    }

    df_working['datetime'] = pd.to_datetime(date_dict).values
    df_working = df_working.sort_values('datetime')
    df_working['time_diff'] = df_working['datetime'].diff()

    print("Does the dataset have inconsistent time intervals?")
    display(df_working['time_diff'].value_counts().sort_index())

    print("----------------------------------------------")

    print("Time gaps analysis plotted")
    plt.figure(figsize=(10,4))
    plt.plot(df_working['datetime'], df_working['time_diff'].dt.total_seconds())
    plt.ylabel('Seconds between records')
    plt.xlabel('Time')
    plt.title('Time Gaps')
    plt.show()

# Provides descriptive statistics and visualization of the target variable, optionally applying a log transformation.
def target_info(dataframe: pd.DataFrame, target: str, transform: Literal["log", "log1p", None] = None):
    """
    - Prints counts and relative frequencies of unique target values.
    - Reports number and percentage of missing values, if any.
    - Plots:
        (1) Histogram and KDE of the raw target.
        (2) Histogram and KDE of the log-transformed target (if specified).
    - Supports both log() and log1p() transforms.

    Parameters:
        dataframe (pd.DataFrame): Input dataset containing the target.
        target (str): Column name of the target variable.
        transform ({"log", "log1p", None}): Optional transform to apply
            - "log" → natural logarithm
            - "log1p" → log(1 + x), safer if zero values exist
            - None → no transform (default)

    Returns:
        None (prints summary and displays plots).
    """
    if target not in dataframe.columns:
        print(f"{target} is not in df!")
        return

    print("Unique target values:")
    target_col = dataframe[target]
    target_values = pd.DataFrame({
        'counts': target_col.value_counts().values, 
        '%': np.round(target_col.value_counts(normalize=True).values,2)
                                },
        index=target_col.value_counts().index
    )

    if target_col.isna().any():
        target_values.loc['NaN'] = [target_col.isna().sum(), np.round(target_col.isna().sum() / len(dataframe) * 100, 2)]


    display(target_values)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,4))

    sns.histplot(target_col, kde=True, color='lightcoral', ax=ax[0])
    ax[0].set_title(f'Distribution of {target}')
    ax[0].set_xlabel('Target Value')
    ax[0].set_ylabel('Density')

    if transform == "log":
        transformed_target = np.log(target_col)
    elif transform == "log1p":
        transformed_target = np.log1p(target_col)
    elif transform == None:
        return
    else:
        print("Invalid 3rd param:", transform)
        return

    sns.histplot(transformed_target, kde=True, color='lightcoral', ax=ax[1])
    ax[1].set_title(f'Distribution of log({target})')
    ax[1].set_xlabel('Target Value')
    ax[1].set_ylabel('Density')

    plt.show()

# Summarizes and visualizes the distribution of a categorical feature.
def get_categoric_info(dataframe: pd.DataFrame, categoric_feature: str):
    """
    - Counts frequency and percentage of each category.
    - Reports missing values (if any).
    - Displays results in a DataFrame.
    - Plots a bar chart showing category percentages.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        categoric_feature (str): Name of the categorical column to analyze.

    Returns:
        None (displays summary table and bar plot).
    """
    feat_col = dataframe[categoric_feature]

    cat_values = pd.DataFrame({'count': feat_col.value_counts().values, 
                          '%': np.round(feat_col.value_counts(normalize=True).values*100, 2),
                          }, index=feat_col.value_counts().index)

    if feat_col.isna().any():
            cat_values.loc['NaN'] = [feat_col.isna().sum(), np.round(feat_col.isna().sum() / len(dataframe) * 100, 2)]

    display(cat_values)

    print("----------------------------")

    fig, ax = plt.subplots(figsize=(20,4))

    sns.barplot(x=cat_values.index, y=cat_values['%'].values, ax=ax)
    ax.set_title('wd values dist')
    ax.set_xlabel('wd')
    ax.set_ylabel('%')

    plt.show()

# Examines how a target variable changes with respect to different time-related features.
def plot_by_time(yaxis_target_name, yaxis_target_col, time_features: dict['col_name': pd.Series], per_plot_width: int = 3, per_plot_height: int = 4, title_font_size: int = 30, tick_font_size: int = 26):
    """
    - Groups the target variable by each time feature (e.g., hour, month).
    - Computes mean target values for each group.
    - Plots bar charts of target means vs. each time feature side by side.

    Parameters:
        yaxis_target_name (str): Label for the target (used in plot titles).
        yaxis_target_col (pd.Series): The target variable values.
        time_features (dict): Dictionary mapping feature name → time feature series.
        per_plot_width (int): Width per subplot (default=3).
        per_plot_height (int): Height per subplot (default=4).
        title_font_size (int): Font size for plot titles.
        tick_font_size (int): Font size for axis ticks.

    Returns:
        None (displays plots).
    """
    time_feat_len = len(time_features)
    fig, ax = plt.subplots(1, time_feat_len, figsize=(per_plot_width * time_feat_len, per_plot_height))

    # Ensure ax is an array even for a single subplot
    if time_feat_len == 1:
        ax = [ax]

    for i, (time_feat_col, time_feat_series) in enumerate(time_features.items()):
        by_filter = yaxis_target_col.groupby(time_feat_series).mean()
        
        ax[i].bar(by_filter.index, by_filter.values)
        ax[i].set_title(f'{yaxis_target_name}: vs {time_feat_col}', fontsize=title_font_size)
        ax[i].set_xlabel(time_feat_col, fontsize=tick_font_size)

        # increasing tick font size
        ax[i].tick_params(axis='x', labelsize=tick_font_size)
        ax[i].tick_params(axis='y', labelsize=tick_font_size)

    plt.tight_layout()
    plt.show()

# Analyzes whether missingness in a target column depends on values of other numerical features.
def plot_missing_vs_features(df, numerical_cols, missing_col, bins=20):
    """
    - Creates a binary indicator (missing/not missing) for the chosen column.
    - Bins each numerical feature (quantile bins by default, with safeguards for dominant values).
    - Calculates fraction of missing values in each bin.
    - Plots bar charts showing how missingness varies with feature values.
    - Returns aggregated results per feature.

    Parameters:
        df (pd.DataFrame): Input dataset.
        numerical_cols (list): List of numerical feature names to test.
        missing_col (str): Column name whose missingness is being analyzed.
        bins (int): Number of bins for quantile discretization (default=20).

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping feature name → DataFrame
            with binned values, fraction missing, and counts.
            Also displays one bar chart per feature.
    """
    df_copy = df.copy()
    miss_col = f"{missing_col}_miss"
    df_copy[miss_col] = df_copy[missing_col].isna().astype(int)

    agg_results = {}

    for col in numerical_cols:
        values = df_copy[col]
        # guard: if all values are NaN, skip
        if values.dropna().empty:
            continue

        # detect dominant single value (exclude NaN)
        vc = values.dropna().value_counts(normalize=True)
        most_common_frac = vc.iloc[0] if not vc.empty else 0

        if most_common_frac > 0.5:
            main_val = vc.index[0]
            mask = values != main_val
            binned = pd.Series(['main'] * len(values), index=values.index)
            if mask.sum() > 0:
                try:
                    binned.loc[mask] = pd.qcut(values[mask], min(bins, mask.sum()), duplicates='drop')
                except ValueError:
                    binned.loc[mask] = pd.cut(values[mask], min(bins, mask.sum()))
        else:
            try:
                binned = pd.qcut(values, bins, duplicates='drop')
            except ValueError:
                binned = pd.cut(values, bins)

        df_copy[f'{col}_bin'] = binned

        # use the boolean miss column for mean
        frac_missing = (df_copy
                        .groupby(f'{col}_bin', observed=True)[miss_col]
                        .mean()
                        .sort_index(key=lambda idx: [(-np.inf if x=='main' else x.left) if hasattr(x, 'left') or x=='main' else 0 for x in idx])
                       )

        # create tick labels
        tick_labels = []
        for interval in frac_missing.index:
            if interval == 'main':
                tick_labels.append(f'{float(main_val):.1f} (majority)')
            else:
                left = getattr(interval, 'left', np.nan)
                right = getattr(interval, 'right', np.nan)
                tick_labels.append(f"{left:.1f}-{right:.1f}")

        # store agg
        agg_df = pd.DataFrame({
            col + '_bin': frac_missing.index.astype(str),
            'frac_missing': frac_missing.values,
            'count': df_copy.groupby(f'{col}_bin', observed=True)[miss_col].count().reindex(frac_missing.index).values
        })
        agg_results[col] = agg_df

        # plot
        plt.figure(figsize=(8,3))
        plt.bar(range(len(frac_missing)), frac_missing.values)
        plt.xticks(range(len(frac_missing)), tick_labels, rotation=45, ha='right')
        plt.ylabel('Fraction missing')
        plt.title(f'{missing_col} missing fraction vs {col}')
        plt.tight_layout()
        plt.show()

# Summarizes missing values in the dataset.
def get_NA_info(data_frame: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    """
    - Counts number and percentage of NaN values for each column (or a subset).
    - Returns a sorted DataFrame with results.

    Parameters:
        data_frame (pd.DataFrame): Input dataset.
        feature_columns (list, optional): Specific columns to check. 
            If None, all columns are checked.

    Returns:
        pd.DataFrame:
            Columns: ['Feature', 'NA Count', 'Total Count', 'NA Percentage'].
            Sorted by NA Count (descending).
        [] if no missing values are found.
    """
    if feature_columns is None:
        feature_columns = data_frame.columns
        
    rows = []

    for feat in feature_columns:
        na_count = data_frame[feat].isna().sum()
        if na_count > 0:
            total_count = data_frame.shape[0]
            na_percent = na_count / total_count * 100

            rows.append((feat, na_count, total_count, na_percent))

    if len(rows) == 0:
        print("No missing values are found in the dataframe")
        return []

    df_result = pd.DataFrame(rows, columns=['Feature', 'NA Count', 'Total Count', 'NA Percentage'])
    df_result.sort_values(by='NA Count', ascending=False, inplace=True, ignore_index=True)

    return df_result

# Visual diagnostic plots of features vs. the target, with optional highlighting of imputed/masked values.
def plot_the_features_against_target(dataframe: pd.DataFrame, features: list, target: str, imputed_masks: dict = None):
    """
    For each feature:
    - Plots raw distribution (histogram + KDE).
    - QQ plot for normality check.
    - Log(1+x)-transformed distribution and QQ plot.
    - Scatterplot/regression line vs. target (or self-plot if feature == target).
    - If imputed_masks are provided, highlights imputed vs. non-imputed points.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        features (list): List of feature names to plot.
        target (str): Target variable name.
        imputed_masks (dict, optional): Dictionary mapping column name → boolean mask 
            (True for imputed values). If provided, imputed values are highlighted in plots.

    Returns:
        None (displays multiple plots for each feature).
    """
    for feat in features:
        print(feat)
        fig, axes = plt.subplots(ncols=5, figsize=(25, 5))

        sns.histplot(dataframe[feat], kde=True, ax=axes[0])
        stats.probplot(dataframe[feat], plot=axes[1])
        sns.histplot(np.log1p(dataframe[feat]), kde=True, ax=axes[2])
        stats.probplot(np.log1p(dataframe[feat]), plot=axes[3])

        if imputed_masks is not None:
            feat_mask = imputed_masks[feat]
            target_mask = imputed_masks[target]
            combined_mask = (feat_mask | target_mask).astype(bool)

            if feat == target:
                plot_df = dataframe[[feat]].copy()
                plot_df["mask"] = combined_mask
                
                sns.scatterplot(
                    x=plot_df[feat],
                    y=plot_df[feat],
                    ax=axes[4],
                    legend=False,
                    s=50,
                    label='Not imputed',
                    marker='o',
                    alpha=0.4
                )
                sns.scatterplot(
                    x=plot_df.loc[plot_df["mask"], feat],
                    y=plot_df.loc[plot_df["mask"], feat],
                    ax=axes[4],
                    legend=False,
                    color='red',
                    s=5,
                    marker='X',
                    label='Imputed'
                )

            else:
                plot_df = dataframe[[feat, target]].copy()
                plot_df["mask"] = combined_mask

                sns.scatterplot(
                    data=plot_df,
                    x=feat,
                    y=target,
                    hue="mask",
                    s=30,
                    palette={False: "#3885BC", True: "red"},
                    ax=axes[4],
                    legend=False,
                    edgecolor='white',
                    linewidths=0.5,
                    alpha = 0.4
                )

                sns.regplot(
                    data=plot_df[~plot_df["mask"]],
                    x=feat,
                    y=target,
                    scatter=False,
                    color='blue',
                    ax=axes[4],
                    scatter_kws={'s': 20, 'edgecolor': 'white', 'linewidths': 0.5, 'alpha':0.4},
                )

        else:
            sns.regplot(
                data=dataframe,
                x=feat,
                y=target,
                ax=axes[4],
                scatter=True,
                scatter_kws={'s': 20, 'edgecolor': 'white', 'linewidths': 0.5, 'alpha':0.4},
                line_kws={'color': 'blue'}
            )

        axes[0].set_title(f'{feat} distribution')
        axes[1].set_title(f'{feat} QQ plot')
        axes[2].set_title(f'log(1+x) {feat} distribution')
        axes[3].set_title(f'log(1+x) {feat} QQ plot')
        axes[4].set_title(f'{feat} visual outliers with regression line')

        plt.show()

# conducts the Little's test on the given dataframe to test the null hypothesis: p-value > 0.05 - data is likely MCAR or MAR / MNAR otherwise
# caveat: textbook option is to include only raw numerical, but if you don’t include categoricals and the whole missingness mechanism depends on them → the test will wrongly suggest MCAR. That’s a known limitation.
def little_mcar_test(data_frame_cont_numeric_raw: pd.DataFrame):
    """
    Performs Little's MCAR test to check if the missing data is Missing Completely At Random (MCAR).
    """
    
    # check if missing values exist
    if data_frame_cont_numeric_raw.isnull().sum().sum() == 0:
        print("No missing values in the DataFrame.")
        return
    
    # Little's MCAR test on the provided dataframe
    mt = MCARTest(method="little")
    p_value = mt.little_mcar_test(data_frame_cont_numeric_raw)

    print(f"p-value = {p_value:.6f}")

    if p_value > 0.05:
        print("Fail to reject the null hypothesis: Data is likely MCAR.")
    else:
        print("Reject the null hypothesis: Data is not MCAR (likely MAR or MNAR).")

# Tests whether missingness in a column is associated with other categorical features.
def chi2_categorical_check(df_raw: pd.DataFrame, categorical_cols: list):
    df = df_raw.copy()
    
    # Identify all columns with missing values in the entire dataframe
    columns_with_nans = df.columns[df.isna().any()].tolist()

    for col in columns_with_nans:
        # Create the missingness indicator flag for the current column
        miss_flag = df[col].isnull().astype(int)

        for other_col in categorical_cols:
            # Skip testing a column against itself
            if col == other_col:
                continue
            
            # Create the contingency table
            tab = pd.crosstab(miss_flag, df[other_col])
            
            # Avoid ValueError if a column has no variation or only one category
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue

            chi2, p, _, _ = stats.chi2_contingency(tab)

            if p < 0.05:
                print(f"Missingness in {col} depends on {other_col} (p={p:.6f})")
            else:
                print(f"Missingness in {col} doesn't depend on {other_col} (p={p:.6f})")


# Standardizes continuous numerical features for modeling.
# returns standardized dataframe, scaler object, df with original stds
def standardize(dataframe: pd.DataFrame, numerical_cols: list, target: str, scaler=None):
    """
    - Removes the target column from features to be scaled.
    - Scales only numerical (non-object) columns with StandardScaler (z-score).
    - Leaves target column unchanged.
    - Returns the scaled DataFrame, the fitted scaler, and scaling factors.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        numerical_cols (list): List of numerical features to standardize.
        target (str): Target variable name (excluded from scaling).
        scaler (StandardScaler, optional): Existing scaler to reuse. If None, fits a new one.

    Returns:
        tuple:
            - df_scaled (pd.DataFrame): DataFrame with scaled numerical features.
            - scaler (StandardScaler): Fitted scaler object.
            - scaler_df_sorted (pd.DataFrame): 1-row DataFrame of scaling factors.
    """
    num_features = sorted(numerical_cols.copy())
    if target in num_features:
        num_features.remove(target)

    if target not in dataframe.columns:
        print(f"{target} is not in df")
        return None, None, None
    
    obj_features = dataframe.select_dtypes(include="object").columns
    if any(col in obj_features for col in num_features):
        print("You can't scale object features, reconsider numerical_cols")
        return None, None, None

    if scaler is None:
        scaler = StandardScaler()

    X = dataframe.drop(columns=target)
    y = dataframe[target]

    # scale only continuous numerical
    X[num_features] = scaler.fit_transform(X[num_features])
    scaler_df_sorted = pd.DataFrame(columns=num_features, data=scaler.scale_.reshape(1, -1))
    
    df_scaled = dataframe.copy()
    df_scaled[num_features] = X[num_features]
    df_scaled[target] = y

    return df_scaled, scaler, scaler_df_sorted[num_features]

# inverts standardization performed by `standardize` function.
def inverse_standardize(dataframe: pd.DataFrame, scaler: StandardScaler, scaler_df_sorted: pd.DataFrame):
    """
    - Applies the scaler.inverse_transform to the standardized columns.
    - Ensures all standardized columns exist in the input DataFrame.

    Parameters:
        dataframe (pd.DataFrame): Input dataset with standardized values.
        scaler (StandardScaler): Fitted StandardScaler used previously.
        scaler_df_sorted (pd.DataFrame): DataFrame with column names that were scaled.

    Returns:
        pd.DataFrame:
            New DataFrame with columns transformed back to original scale.
        None if required columns are missing.
    """
    inversed_df = dataframe.copy()
    standardized_cols = scaler_df_sorted.columns

    missing_cols = [c for c in standardized_cols if c not in inversed_df.columns]
    if missing_cols:
        print(f"Error: standardized columns {missing_cols} not in df")
        return None

    inversed_df[standardized_cols] = scaler.inverse_transform(inversed_df[standardized_cols])

    return inversed_df

# Visualizes correlations between selected features as a heatmap.
def plot_correlations(data_frame: pd.DataFrame, features_to_plot: list, title: str, threshold: float = 0.0):
    """
    - Computes correlation matrix among selected features.
    - Filters to features with at least one correlation ≥ threshold (by abs value).
    - Plots a heatmap with annotated correlation values.
    - Ignores self-correlations.

    Parameters:
        data_frame (pd.DataFrame): Input dataset.
        features_to_plot (list): Feature names to include in correlation matrix.
        title (str): Title for the heatmap.
        threshold (float, default=0.0): Minimum absolute correlation to display.

    Returns:
        None (displays heatmap).
    """
    correlations = data_frame[features_to_plot].corr()
    
    # Find features that have any strong correlation (excluding self-correlation)
    mask = (abs(correlations) >= threshold) & (~np.eye(len(correlations), dtype=bool))
    keep_features = correlations.columns[mask.any(axis=1)]

    # Filter to those features
    filtered = correlations.loc[keep_features, keep_features]

    if filtered.empty:
        print(f"No entries to plot for a threshold of {threshold}")
        return

    np.fill_diagonal(filtered.values, np.nan)

    plt.figure(figsize=(15, 15))
    sns.heatmap(filtered, annot=True, fmt=".2f")

    if threshold != 0.0:
        title = title + f' (|corr| ≥ {threshold})'

    plt.title(title)
    plt.show()

    return correlations

# Checks correlations of one feature with others, marking whether correlated features contain missing values.
def get_corr_null_info_for_a_feat(dataframe: pd.DataFrame, feature: str, corrs: pd.DataFrame, num_features: list):
    """
    - Extracts correlations of a specific feature against numerical features.
    - Drops self-correlation.
    - Builds DataFrame with absolute correlations and null indicators.
    - Sorts by descending absolute correlation.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        feature (str): Feature name to analyze.
        corrs (pd.DataFrame): Precomputed correlation matrix.
        num_features (list): List of numerical feature names.

    Returns:
        pd.DataFrame: Features with correlation strength and null-flag.
    """
    # Step 2: Restrict the correlation matrix properly
    correlations = corrs.loc[num_features]
    
    if feature not in correlations.columns:
        print(f"Feature '{feature}' not found in correlation matrix.")
        return

    # Step 3: Drop self-correlation
    feat_corr = correlations[feature].drop(labels=[feature], errors='ignore')

    # Step 4: Build result DataFrame
    result = pd.DataFrame({
        'abs_corr': feat_corr.abs(),
        'has_nulls': dataframe[feat_corr.index].isna().any().values
    })

    # Step 5: Sort and display
    result = result[result['abs_corr'] < 1.0].sort_values(by='abs_corr', ascending=False)
    return result

# Calculates Variance Inflation Factor (VIF) to assess multicollinearity.; ensures the target isn't included, sorts the result with `ascending = False` by default
def perform_VIF(data_frame_no_nulls: pd.DataFrame, encoded_predictors: list, target: str, ascending: str = False):
    """
    - Builds regression matrix for predictors (excluding target).
    - Computes VIF for each predictor.
    - Higher VIF means stronger multicollinearity.

    Parameters:
        data_frame_no_nulls (pd.DataFrame): Clean dataset without nulls.
        encoded_predictors (list): List of predictor names.
        target (str): Target variable name (excluded from predictors).
        ascending (bool, default=False): Sort order for VIF values.

    Returns:
        pd.DataFrame: Features with their VIF scores, sorted.
    """
    if target in encoded_predictors:
        encoded_predictors.remove(target)

    X = data_frame_no_nulls[encoded_predictors]
    X = add_constant(X)

    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = vif_df[vif_df["feature"] != "const"]
    
    return vif_df.sort_values(by="VIF", ascending=ascending)

# Iteratively identifies and suggests predictors to drop to reduce multicollinearity (VIF below threshold).
def get_VIF_features_to_drop(dataframe_scaled_numerical: pd.DataFrame, target: str, threshold: float = 5.0):
    """
    - Runs VIF analysis.
    - Iteratively removes the feature with highest VIF ≥ threshold.
    - Records max VIF after each drop.

    Parameters:
        dataframe_scaled_numerical (pd.DataFrame): Scaled numerical features.
        target (str): Target variable name (excluded).
        threshold (float, default=5.0): VIF cutoff.

    Returns:
        list[str]: Steps showing max VIF before/after each drop.
    """
    vif_df = perform_VIF(dataframe_scaled_numerical, list(dataframe_scaled_numerical.columns), target)
    max_val = vif_df['VIF'].max()
    to_drop = [f"before dropping anything: {max_val}"]

    while max_val >= threshold:
        max_val = vif_df['VIF'].max()
        feat = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        
        if max_val >= threshold:
            dataframe_scaled_numerical = dataframe_scaled_numerical.drop(columns=[feat])
            vif_df = perform_VIF(dataframe_scaled_numerical, list(dataframe_scaled_numerical.columns), target)
            to_drop.append(f"after dropping {feat}: {vif_df['VIF'].max()}")
        else:
            return to_drop
        
    return to_drop

# Estimates how many outliers would be capped under specified winsorization rules.
def pick_winsorization_thresholds(dataframe: pd.DataFrame, features_with_boundaries: dict):
    """
    - For each feature, applies lower/upper quantile cutoffs.
    - Marks values outside thresholds as outliers.
    - Reports % of values flagged for capping.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        features_with_boundaries (dict): Feature → (lower_quantile, upper_quantile).

    Returns:
        pd.DataFrame: Boolean mask for flagged outliers per feature.
    """
    df = dataframe.copy()
    winsorized_masks = pd.DataFrame(np.full((len(df), len(features_with_boundaries)), False), columns=features_with_boundaries.keys(), index=df.index)

    for feat, boundaries in features_with_boundaries.items():
        lower_boundary = boundaries[0]
        upper_boundary = boundaries[1]

        X = df[feat]

        lower = X.quantile(lower_boundary)
        upper = X.quantile(1 - upper_boundary)
        mask = (X < lower) | (X > upper)

        winsorized_masks[feat] = mask

        upper_sum = (X > upper).sum()
        lower_sum = (X < lower).sum()

        outliers_upper = upper_sum / len(X)
        outliers_lower = lower_sum / len(X)

        print(f"{outliers_upper + outliers_lower:.2%} ({upper_sum} upper and {lower_sum} lower) of {feat} would be capped")

    return winsorized_masks

# Runs the Breusch-Pagan test to assess heteroscedasticity (dependent on predictors).
def breusch_pagan_test(residuals, X_with_const):
    # Run Breusch-Pagan test
    lagrange_mult, p_val, f_val, fp_val = het_breuschpagan(residuals, X_with_const)

    if p_val > 0.05:
        print("Fail to reject the null hypothesis: No Heteroscedasticity.")
    else:
        print("Reject the null hypothesis: Heteroscedasticity is present.")

    return pd.DataFrame(columns=['Lagrange multiplier statistic', 'R2', 'p-value', 'f-value', 'f p-value'], data=[[lagrange_mult, lagrange_mult/len(residuals), p_val, f_val, fp_val]]).style.hide(axis="index")

# Runs the Breusch-Pagan test to assess heteroscedasticity (dependent on time).
def het_arch_test(residuals, n_lags):
    # Run Breusch-Pagan test
    lagrange_mult, p_val, f_val, fp_val = het_arch(residuals, n_lags)

    if p_val > 0.05:
        print("Fail to reject the null hypothesis: No Heteroscedasticity.")
    else:
        print("Reject the null hypothesis: Heteroscedasticity is present.")

    return pd.DataFrame(columns=['Lagrange multiplier statistic', 'R2', 'p-value', 'f-value', 'f p-value'], data=[[lagrange_mult, lagrange_mult/len(residuals), p_val, f_val, fp_val]]).style.hide(axis="index")

# Detects perfect multicollinearity using matrix rank.; ensures the target is dropped; returns True if perfect multicollinearity detected or False otherwise; also returns the matrix rank
def detect_perfect_multicollinearity_via_rank(dataframe: pd.DataFrame, target: str, verbose: bool = False):
    """
    - Selects numerical predictors (excluding target).
    - Computes matrix rank.
    - Compares rank to number of features.
    - If rank < n_features → perfect multicollinearity present.

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        target (str): Target variable name (excluded from X).
        verbose (bool, default=False): Print details if True.

    Returns:
        tuple[bool, int]:
            - Flag indicating presence of perfect multicollinearity.
            - Rank of the predictor matrix.
    """
    dataframe = dataframe.select_dtypes(include="number")
    if target in dataframe.columns:
        X = dataframe.drop(columns=[target]).dropna()
    else:
        X = dataframe.dropna()
    rank = matrix_rank(X.values)
    n_features = X.shape[1]

    if verbose:
        print(f"Matrix rank: {rank} / {n_features} features")

    if rank < n_features:
        if verbose:
            print("Perfect multicollinearity detected.")

        return True, rank
    
    return False, rank

# tries to find the smallest possible subset of features to drop that resolves perfect multicollinearity in one go; prioritizes feature-engineered predictors 
# (i.e., sorts by moving features with underscore to the top; I tend to call derived predictors according to that standard)
def detect_features_to_restore_full_rank(dataframe: pd.DataFrame, target: str):
    """
    Identifies minimal set of features to drop in order to restore full column rank
    (i.e., eliminate perfect multicollinearity).

    - Computes initial rank of feature matrix (excluding target).
    - If rank < n_features → perfect multicollinearity exists.
    - Iterates through feature subsets to find smallest set to drop that restores independence.
    - Prioritizes engineered features (features with "_").

    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        target (str): Target variable (excluded from analysis).

    Returns:
        list[str] | str:
            - Names of features to drop if solution exists.
            - "No perfect multicollinearity detected" if none present.
            - "Could not resolve..." if no subset fixes rank.
    """
    features = [col for col in dataframe.columns if col != target] # exclude the target (we can't drop it)
    features.sort(key=lambda x: not ("_" in x))  # prioritize dropping feature-engineered features

    initial_rank = matrix_rank(dataframe[features].dropna().values) # drop nulls (just in case) and calculate the rank
    n_features = len(features)

    # If there's no perfect multicollinearity, nothing to drop
    if initial_rank == n_features:
        return "No perfect multicollinearity detected"

    # Try all combinations of 1 up to n features to remove
    for k in range(1, n_features + 1):
        for combo in combinations(features, k):
            # Keep features not in the current combo
            candidate_features = [f for f in features if f not in combo]

            # Check if the remaining features are linearly independent separately
            candidate_rank = matrix_rank(dataframe[candidate_features].values)
            if candidate_rank == len(candidate_features):
                return list(combo)

    # If no subset fixes the issue
    return "Could not resolve perfect multicollinearity by dropping any subset of features."

# Generates a boolean mask simulating missingness across features for imputation testing.
def get_a_mask_for_features(data_frame: pd.DataFrame, 
                            cont_num_features_to_mask: list[str]=[], 
                            ord_categorical_features_to_mask: list[str]=[], 
                            nominal_groups_to_mask: dict[str, list[str]]=[],
                            rand_seed: int=42, 
                            missing_fraction: float = 0.05) -> pd.DataFrame:

    # a copy not to modify the original
    df_work = data_frame.copy()
    # rng init for reproducibility
    rng = np.random.default_rng(seed=rand_seed)
    # calculate the amount of artificial nulls per feature (minimum 1)
    n = len(df_work)
    missing_count = max(1, int(n * missing_fraction)) 

    # bring all the columns together
    all_cols = cont_num_features_to_mask + ord_categorical_features_to_mask 
    if nominal_groups_to_mask:
        all_cols += [c for g in nominal_groups_to_mask.values() for c in g]
    all_cols = sorted(set(all_cols))

    # initial mask
    df_mask = pd.DataFrame(False, index=df_work.index, columns=all_cols)

    # create artificial masks for continuous + ordinal
    for col in cont_num_features_to_mask + ord_categorical_features_to_mask:
        # sample unique indexes
        idx = rng.choice(a=n, size=missing_count, replace=False) # replace = 'False' because 'a' cannot be selected multiple times, we don't want to mask same values repeatedly
        df_mask.iloc[idx, df_mask.columns.get_loc(col)] = True

    # for nominals, we have to treat each one-hot-encoded group as one feature
    # + to account for the caveat where a single line must all contain NaN if it's absent.
    if nominal_groups_to_mask:
        for _, group_cols in nominal_groups_to_mask.items():
            idx = rng.choice(n, size=missing_count, replace=False)
            df_mask.loc[idx, group_cols] = True

    return df_mask

# Evaluates imputation quality by applying controlled missingness and comparing imputed values against ground truth.
def test_imputer_scores(
    imputer: IterativeImputer,
    data_frame: pd.DataFrame, # unless a slice is provided, the whole dataframe is used
    numericals_and_ordinals_to_impute: list[str], # only columns to impute (must all be in the data_frame; can be both num or ord or either)
    all_dataframe_nominal_groups: dict[str, list[str]], # all nominals from data_frame (both missing and not)
    mask: pd.DataFrame, # created with get_a_mask_for_features, where True means the value is missing
    mode: Literal["regression", "classification"] = "regression",
    verbose: bool = True
) -> pd.DataFrame:
    
    def compute_metrics(true_vals, pred_vals, col_name):
        if mode == 'regression':
            mae = mean_absolute_error(true_vals, pred_vals)
            rmse = root_mean_squared_error(true_vals, pred_vals)
            r2 = r2_score(true_vals, pred_vals)

            return [round(mae,4), round(rmse,4), round(r2,4),
                    round(data_frame[col_name].min(), 3), round(data_frame[col_name].max(), 3),
                    round(data_frame[col_name].std(),4)]
        elif mode == "classification":
            acc = accuracy_score(true_vals, pred_vals)
            f1 = f1_score(true_vals, pred_vals, average="macro")

            return [round(acc, 4), round(f1, 4)]

    # Which columns to impute (flatten groups)
    cols_to_impute = numericals_and_ordinals_to_impute + [c for g in all_dataframe_nominal_groups.values() for c in g]

    if set(cols_to_impute) - set(mask.columns.tolist()):
        print("Please, check your `cols_to_impute`, not all of the specified columns are in the mask")
        return
    
    if set(cols_to_impute) - set(data_frame.columns.tolist()):
        print("Please, check your `cols_to_impute`, not all of the specified columns are in the dataframe")
        return

    if mode != "regression" and mode != "classification":
        print(f"the 'mode' parameter can be either 'regression' or 'classification', {mode} given")
        return
    
    if not data_frame[cols_to_impute].isna().any().any():
        print("No missing values in num_cols_to_impute, nothing to test.")
        return

    print(f"(Evaluation) The amount of masked NaNs for each column to impute: {(mask == True).sum().values[0]} / {mask.shape[0]} ({((mask == True).sum().values[0] / mask.shape[0])* 100:.3f}%)\nCols to impute: {cols_to_impute}")

    if mode == "regression":
        result = pd.DataFrame(columns=["MAE", "RMSE", "R2", 'min', 'max', 'col_std'], index=[])
    else:
        result = pd.DataFrame(columns=["Accuracy", "f1-score"], index=[])

    features_ordered = sorted(set(data_frame.columns))
    df_train = data_frame.copy()
    df_eval = data_frame.copy()

    # apply artificial mask to eval only
    df_eval[mask] = np.nan

    # Fit the imputer on train to let the imputer memorize the nulls and fit them with the initial strategy
    imputer.fit(df_train[features_ordered])

    # Impute the missing values in the eval dataframe after training
    imputed_array = imputer.transform(df_eval[features_ordered])

    # imputed all columns, but applying only the specified cols_to_impute
    df_imputed = pd.DataFrame(imputed_array, columns=features_ordered, index=df_eval.index)
    df_eval[cols_to_impute] = df_imputed[cols_to_impute]

    # ---- Evaluate numericals and ordinals ----
    for col in numericals_and_ordinals_to_impute:
        curr_mask = mask[col] # mask for curr null col
        
        if curr_mask.any():
            true_vals = data_frame.loc[curr_mask, col].dropna() # drop original nulls
            pred_vals = df_eval.loc[curr_mask, col].loc[true_vals.index] # get the predicted values for existent values
            result.loc[col] = compute_metrics(true_vals, pred_vals, col) # evaluate with ground truth

    # ---- Evaluate nominals ----
    for group_name, cols in all_dataframe_nominal_groups.items():
        curr_mask = mask[cols].any(axis=1)

        if curr_mask.any():
            true_vals = data_frame.loc[curr_mask, cols].idxmax(axis=1)
            pred_vals = df_eval.loc[curr_mask, cols].idxmax(axis=1)
            result.loc[group_name] = compute_metrics(true_vals, pred_vals, group_name)

    if verbose:
        if mode == 'regression':
            mean_mae = result['MAE'].astype(float).mean()
            mean_rmse = result['RMSE'].astype(float).mean()

            print(f"Mean MAE: {mean_mae:.4f}, Mean RMSE: {mean_rmse:.4f}")
        else:
            mean_acc = result['Accuracy'].astype(float).mean()
            mean_f1 = result['f1-score'].astype(float).mean()

            print(f"Mean Accuracy: {mean_acc:.4f}, Mean f1-score: {mean_f1:.4f}")

    return result.sort_index()

# Visualizes relationship between two features, with missingness highlighted.
def plot_scatter_relationship_colored_by_missingness(data_frame: pd.DataFrame, feature1: str, feature2: str, target: pd.Series):
    """
    - Plots scatter of feature1 vs feature2.
    - Colors points by binary missingness indicator (0/1).
    - Adds colorbar legend.

    Parameters:
        data_frame (pd.DataFrame): Input dataset.
        feature1 (str): First variable for scatterplot.
        feature2 (str): Second variable.
        target (pd.Series): Missingness indicator (0 = observed, 1 = missing).

    Returns:
        None (displays plot).
    """
    
    # Scatter plot: colored by missingness
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        data_frame[feature1],
        data_frame[feature2],
        c=target,  # 0/1 missingness
        cmap="coolwarm",
        alpha=0.5
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Missingness (1 = missing, 0 = not missing)")

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"Relationship between {feature1} and {feature2},\ncolored by Missingness")

    plt.show()

# Classification-based greedy forward lag-set selector (utilizes GEE with custom weights)
def pick_ljungbox_lags_for_classification(
        data_frame: pd.DataFrame,
        candidate_features: np.ndarray, # sensible features
        target_var: str, # classification target
        lag_degree: int = 1, # lag order
        lb_horizon: int = 24, # horizon for the Ljung-Box (24 by default. Alternatives: 24 for a day, 168 for hourly)
        max_combo_size: int = None, # combination size limit to try; None means no limit
        weights: np.array = None # custom target weights; may be None
    ):
    """
    - Fits a logistic model for each candidate combo of lagged features.
    - Computes residual-like series resid = y - p_hat.
    - Runs Ljung-Box joint test up to lb_horizon.
    - Tracks auc.
    - Tracks VIF.
    - Tracks acf computed via weighed Pearson (with custom weights if specified).
    Returns DataFrame sorted
    """

    df = data_frame.copy()
    if max_combo_size is None:
        max_combo_size = len(candidate_features)

    n_features = len(candidate_features)
    total_iters = total_combos(n_features, max_combo_size)

    print("Expected number of combinations:", total_iters)

    # precompute lagged columns to only drop first row once
    lagged = {}
    for feat in candidate_features:
        lagged[feat] = df[feat].shift(lag_degree).iloc[lag_degree:].reset_index(drop=True)
    # drop first rows globally to align with lagged values
    df_aligned = df.iloc[lag_degree:].reset_index(drop=True)
    acf_lagged_label = f"acf_{lag_degree}"

    # sanity check
    for feat in candidate_features:
        if df[feat].isna().any():
            print(f"Feature {feat} has null(s). Please, replace with a proper central tendency first")
            return

    results = []

    # helper to fit logistic and compute resid and lb p-value
    def fit_and_test_with_gee(eval_df, y_col, weights_val):
        # subset weights if provided
        if weights_val is not None:
            w = np.asarray(weights_val)[eval_df.index]
        else:
            w = np.ones(len(eval_df)) # same weights for everything if the custom not provided

        # try to fit the model
        try:
            X = sm.add_constant(eval_df.drop(columns=[y_col]))
            y = eval_df[y_col].values
            model = sm.GEE(endog=y, exog=X, groups=np.ones(len(y)),
                        family=sm.families.Binomial(), # binary target
                        cov_struct=Independence(), # assume rows are independent
                        weights = w
                    )
            res = model.fit()
            p_hat = res.fittedvalues # train predictions
        except Exception as e:
            print(f"'fit_and_test_with_gee' fit exception: {e}")
            return None

        resid = (y - p_hat).astype(float) # aka classification residuals

        # try to track the acorr_ljungbox
        try:
            lb_df = acorr_ljungbox(resid, lags=[lb_horizon], return_df=True)
            lb_pvalue = float(lb_df['lb_pvalue'].iloc[0])
            lb_stat   = float(lb_df['lb_stat'].iloc[0])
        except Exception as e:
            print(f"'acorr_ljungbox' test exception: {e}")
            lb_pvalue = np.nan
            lb_stat = np.nan

        # try to track auc score
        try:
            auc = roc_auc_score(y, p_hat)
        except Exception as e:
            print(f"'roc_auc_score' exception: {e}")
            auc = np.nan

        # compute residuals: raw + weighted Pearson (not using res.resid_pearson specifically to calculate weighted Pearson - for custom weights)
        eps = 1e-12 # to account for zero division
        p_hat = np.clip(p_hat, eps, 1 - eps) # replace zeroes with epsilon
        pearson_resid_weighted = resid / np.sqrt(w * p_hat * (1 - p_hat) + eps)

        # acf of weighted Pearson residuals
        try:
            acf_values = acf(pearson_resid_weighted, nlags=lag_degree, 
                             fft=False) # use direct/brute-force method (slower for large data, but 100% precise)
            acf_val = float(acf_values[lag_degree]) if len(acf_values) > 1 else np.nan # extract the chosen lag
        except Exception as e:
            print(f"'acf' Exception: {e}")
            acf_val = np.nan

        return {'lb_pvalue': lb_pvalue, 'lb_stat': lb_stat, 'auc': auc, acf_lagged_label: acf_val}

    # iterate combos
    counter = 0
    for r in range(0, max_combo_size + 1):
        for combo in itertools.combinations(candidate_features, r):
            counter += 1
            if counter % 10 == 0:
                print(f"Progress: {counter}/{total_iters} ({100*counter/total_iters:.1f}%)")

            if len(combo) == 0:
                continue
            eval_df = df_aligned.copy()
            # add candidate lag columns
            for feat in combo:
                lag_col = f"{feat}_lag_{lag_degree}"
                eval_df[lag_col] = lagged[feat].values

            # run VIF
            try:
                max_vif_col, max_vif_score = perform_VIF(eval_df, list(eval_df.columns), target_var).max().values
            except Exception as e:
                print(f"'perform_VIF' Exception: {e}")
                max_vif_col, max_vif_score = (np.nan, np.nan)

            # run fit + metrics
            test_result = fit_and_test_with_gee(eval_df, target_var, weights)
            if test_result is None:
                # record failure with NaNs so the combo isn't chosen accidentally
                results.append({
                    'combo': combo,
                    'lb_pvalue': np.nan,
                    'lb_stat': np.nan,
                    acf_lagged_label: np.nan,
                    'auc': np.nan,
                    'cand_len': len(combo),
                    'max_vif_col': max_vif_col,
                    'max_vif_score': max_vif_score,
                })
                continue

            results.append({
                'combo': combo,
                'lb_pvalue': test_result['lb_pvalue'],
                'lb_stat': test_result['lb_stat'],
                acf_lagged_label: test_result[acf_lagged_label],
                'auc': test_result['auc'],
                'cand_len': len(combo),
                'max_vif_col': max_vif_col,
                'max_vif_score': max_vif_score,
            })

    res_df = pd.DataFrame(results)
    res_df['abs_'+acf_lagged_label] = res_df[acf_lagged_label].abs()

    # Sort
    res_df = res_df.sort_values(
        by=['cand_len', 'lb_pvalue', 'abs_'+acf_lagged_label, 'max_vif_score', 'auc'],
        ascending=[True, False, True, True, False]  # true - minimum, false - maximum
    ).reset_index(drop=True)

    return res_df

# regression version of `pick_ljungbox_lags_for_classification`
def pick_ljungbox_lags_for_regression(
        data_frame: pd.DataFrame,
        candidate_features: np.ndarray,
        target_var: str,
        lag_degree: int = 1,
        max_combo_size: int = None,
        rand_seed: int = 42,
        log_scale: bool = False # if the target is log transformed and we need to rescale the metrics
    ):
    """
    Regression version.
    - Fits OLS for each candidate combo of lagged features.
    - Computes residuals y - y_hat.
    - Runs Ljung-Box joint test.
    - Keeps VIF diagnostics.
    """

    df = data_frame.copy()

    if max_combo_size is None:
        max_combo_size = len(candidate_features)

    n_features = len(candidate_features)
    total_iters = total_combos(n_features, max_combo_size)

    print("Expected number of combinations:", total_iters)

    # precompute lagged columns
    lagged = {}
    for feat in candidate_features:
        lagged[feat] = df[feat].shift(lag_degree).iloc[lag_degree:].reset_index(drop=True)
    df_aligned = df.iloc[lag_degree:].reset_index(drop=True)
    acf_lagged_label = f"acf_{lag_degree}"

    results = []

    # helper: fit OLS + Ljung-Box
    def fit_and_test(eval_df, y_col):
        eval_df = eval_df.dropna()
        if eval_df.shape[0] < 30:
            return None

        X = sm.add_constant(eval_df.drop(columns=[y_col]))
        y = eval_df[y_col]

        try:
            model = sm.OLS(y, X).fit()
            y_hat = model.fittedvalues
            model = lgb.LGBMRegressor(
                    objective='regression',
                    boosting_type='gbdt',
                    max_depth=20,
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=200,
                    min_data_in_leaf=50,
                    random_state=rand_seed,
                    verbose=-1,
                    n_jobs=os.cpu_count()-1
                    )

            model.fit(X, y)
            y_hat = model.predict(X)

            resid = y - y_hat

            # Back-transform for metrics if needed
            if log_scale:
                y_true_bt = np.exp(y)
                y_hat_bt = np.exp(y_hat)
            else:
                y_true_bt = y
                y_hat_bt = y_hat

            mae = mean_absolute_error(y_true_bt, y_hat_bt)
            rmse = root_mean_squared_error(y_true_bt, y_hat_bt)

        except Exception as e:
            print(e)
            return None

        try:
            lb_df = acorr_ljungbox(resid, lags=[24], return_df=True)
            lb_pvalue = float(lb_df['lb_pvalue'].iloc[0])
            lb_stat   = float(lb_df['lb_stat'].iloc[0])
        except Exception as e:
            print(e)
            lb_pvalue = float('nan')
            lb_stat = float('nan')

        try:
            acf_values = acf(resid, nlags=1, fft=False, missing='conservative')
            acf_val = float(acf_values[1]) if len(acf_values) > 1 else np.nan
        except Exception as e:
            print(e)
            acf_val = np.nan

        return {
            'lb_pvalue': lb_pvalue, 
            'lb_stat': lb_stat, 
            acf_lagged_label: acf_val,
            'MAE': mae,
            'RMSE': rmse
        }

    # iterate combos
    counter = 0
    for r in range(0, max_combo_size + 1):
        for combo in itertools.combinations(candidate_features, r):
            counter += 1
            if counter % 10 == 0:
                print(f"Progress: {counter}/{total_iters} ({100*counter/total_iters:.1f}%)")

            if len(combo) == 0:
                continue
            eval_df = df_aligned.copy()
            for feat in combo:
                lag_col = f"{feat}_lag_{lag_degree}"
                eval_df[lag_col] = lagged[feat].values

            try:
                max_vif_col, max_vif_score = perform_VIF(eval_df, list(eval_df.columns), target_var).max().values
            except Exception as e:
                print(e)
                max_vif_col, max_vif_score = (None, float('nan'))

            test_result = fit_and_test(eval_df, target_var)
            if test_result is None:
                results.append({
                    'combo': combo,
                    'lb_pvalue': float('nan'),
                    'lb_stat': float('nan'),
                    acf_lagged_label: float('nan'),
                    'MAE': float('nan'),
                    'RMSE': float('nan'),
                    'cand_len': len(combo),
                    'max_vif_col': max_vif_col,
                    'max_vif_score': max_vif_score,
                })
                continue

            results.append({
                'combo': combo,
                'lb_pvalue': test_result['lb_pvalue'],
                'lb_stat': test_result['lb_stat'],
                acf_lagged_label: test_result[acf_lagged_label],
                'MAE': test_result['MAE'],
                'RMSE': test_result['RMSE'],
                'cand_len': len(combo),
                'max_vif_col': max_vif_col,
                'max_vif_score': max_vif_score,
            })

    res_df = pd.DataFrame(results)
    res_df['abs_'+acf_lagged_label] = res_df[acf_lagged_label].abs().fillna(999)

    res_df = res_df.sort_values(
        by=['lb_pvalue', 'abs_'+acf_lagged_label, 'max_vif_score', 'cand_len', 'MAE', 'RMSE'],
        ascending=[False, True, True, True, True, True]
    ).reset_index(drop=True)

    return res_df

class CategoricalImputer(BaseEstimator, TransformerMixin):
    # ordinal_cols and nominal_cols are all the encoded categoricals (the ones that should be imputed AND the ones that act as predictors); can accept numerical predictors as well (just leave them in the df during fit); does not impute columns that are 100% missing
    def __init__(self, ordinal_cols_to_impute:list, nominal_cols_to_impute:dict[str, list[str]], estimator=None, strategy: Literal["most_frequent", "stratified"]=None, random_state:int=42):

        self.ordinal_cols_to_impute = ordinal_cols_to_impute if isinstance(ordinal_cols_to_impute, list) else [ordinal_cols_to_impute]
        self.nominal_cols_to_impute = nominal_cols_to_impute if nominal_cols_to_impute is not None else {}
        self.strategy = strategy
        self.random_state = random_state
        self.estimator = estimator
        self.models_ = {}
        self.feature_names_in_ = None

    def fit(self, X):
        X_copy = X.copy() # X_df_train
        self.feature_names_in_ = X_copy.columns.to_numpy()  # Save the column names (feature_names_in_ is sklearn convention)
        self.models_ = {} # overwrites any previously trained {col:model}

        if self.estimator is not None: # if the classifier estimator is provided explicitly, use it
            est = self.estimator
        elif self.strategy is not None: # if no, demand 'strategy' and if it's among the allowed values, use it
            if self.strategy in ["most_frequent", "stratified"]:
                est = DummyClassifier(strategy=self.strategy, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        else:
            raise ValueError(f"Either specify the 'strategy' or give the 'estimator'")

        # ---- Handle ordinals ----
        for col in self.ordinal_cols_to_impute:
            mask_observed = X_copy[X_copy[col].notna()] # df slice with only rows where this column is NOT missing
        
            if mask_observed.shape[0] == 0: # skip if all missing
                continue

            est = clone(est) # get a fresh, untrained instance of the estimator with the same hyperparameters (to reset the fit it had already been fitted)
            est.fit(mask_observed.drop(columns=[col]), # leave all other columns but the current one 
                    mask_observed[col]) # the current one is target
            self.models_[col] = est

        # ---- Handle nominals ----
        for group_name, cols in self.nominal_cols_to_impute.items():
            mask_observed = X_copy[X_copy[cols].notna()] # if all values are non-nulls in the current group

            if mask_observed.shape[0] == 0: # skip if all missing
                continue

            y_group = X.loc[mask_observed, cols].idxmax(axis=1)  # identify the col with '1' (collapse one-hot)
            est = clone(est)
            est.fit(X.loc[mask_observed].drop(columns=cols), y_group)
            self.models_[group_name] = (est, cols)

        return self

    def transform(self, X):
        X_copy = X.copy()

        # ---- Ordinals ----
        for col in self.ordinal_cols_to_impute:
            mask = X_copy[col].isna()
            if mask.any():
                preds = self.models_[col].predict(X_copy.loc[mask].drop(columns=[col]))
                X_copy.loc[mask, col] = preds

        # ---- Nominals ----
        for group_name, group_cols in self.nominal_cols_to_impute.items():
            model = self.models_.get(group_name)
            mask = X[group_cols].isna().any(axis=1)
            if mask.any():
                preds = model.predict(X.loc[mask].drop(columns=group_cols))

                # Set the predicted class to 1, others to 0
                for col in group_cols:
                    X.loc[mask, col] = (preds == col).astype(int)

        # return numpy array, consistent with sklearn imputers
        return X_copy.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.array(input_features)

# Applies a pre-fitted imputer to fill missing values in selected columns.
def impute_columns(
        fitted_imputer: IterativeImputer | CategoricalImputer, 
        dataframe: pd.DataFrame, 
        numericals_and_ordinals_to_impute: list[str],
        nominal_groups_to_impute: dict[str, list[str]] = None):

    df_imputed = dataframe.copy()
    columns_during_fit = fitted_imputer.feature_names_in_

    # Transform all columns
    df_imputed_array = fitted_imputer.transform(df_imputed[columns_during_fit])
    df_imputed_full = pd.DataFrame(df_imputed_array, columns=columns_during_fit, index=df_imputed.index)

    # Ordinals: copy directly
    if numericals_and_ordinals_to_impute:
        df_imputed[numericals_and_ordinals_to_impute] = df_imputed_full[numericals_and_ordinals_to_impute]

    # Nominals: collapse group names and assign one-hot columns
    if nominal_groups_to_impute:
        for group_name, cols in nominal_groups_to_impute.items():
            df_imputed[cols] = df_imputed_full[cols]
    
    return df_imputed

# Residualizes a single feature on a list or another features.
def residualize(dataframe_train_encoded: pd.DataFrame, 
                dataframe_test_encoded: pd.DataFrame, 
                feat_to_residualize: str, 
                feat_to_residualize_on: np.ndarray,
                dot_tol: float = 1e-8): # absolute dot-product tolerance,
    
    """
    - Fits linear regression of `feat_to_residualize` on controls.
    - Computes residuals (difference between actual and predicted).
    - Ensures residuals are orthogonal to controls (dot-product check).
    - Returns residualized features for train/test.

    Parameters:
        dataframe_train_encoded (pd.DataFrame): Training set (encoded, no NaNs).
        dataframe_test_encoded (pd.DataFrame): Test set (encoded).
        feat_to_residualize (str): Column to orthogonalize.
        feat_to_residualize_on (np.ndarray): Control feature names.
        dot_tol (float): Numerical tolerance for orthogonality check.

    Returns:
        tuple:
            - dot_products (pd.Series): Dot products between residual and controls.
            - resid_feat_col_train (pd.Series): Residualized feature (train).
            - resid_feat_col_test (pd.Series): Residualized feature (test).
    """
    
    # copies for temporary median/mode imputation
    df_train_enc = dataframe_train_encoded.copy()
    df_test_enc = dataframe_test_encoded.copy()

    # Fit residualizer on TRAINING only
    regressor = LinearRegression().fit(df_train_enc[feat_to_residualize_on], df_train_enc[feat_to_residualize])

    # Create TEMP_residual in TRAINING and TEST
    resid_feat_col_train = df_train_enc[feat_to_residualize] - regressor.predict(df_train_enc[feat_to_residualize_on])
    resid_feat_col_test = df_test_enc[feat_to_residualize] - regressor.predict(df_test_enc[feat_to_residualize_on])

    # centered versions for orthogonality check (train)
    resid_centered = resid_feat_col_train - resid_feat_col_train.mean()
    controls_centered = df_train_enc[feat_to_residualize_on] - df_train_enc[feat_to_residualize_on].mean()

    # Sanity check. Orthogonality: centered dot product between feat_to_residualize and feat_to_residualize_on (train)
    # dot products: one scalar per control = control_centered · resid_centered
    dot_products = pd.Series(controls_centered.T.values.dot(resid_centered.values),
                             index=controls_centered.columns)

    ok_dot = (dot_products.abs() < dot_tol).all()

    if ok_dot:
        print("Residualizing succeeded: orthogonality achieved (within numerical tolerance).")
    else:
        print("Warning: orthogonality not fully met. Inspect dot_products and correlations.")

    return dot_products, resid_feat_col_train, resid_feat_col_test

# Helper function to calculate the max amount of combinations.
def total_combos(n_features: int, max_combo_size: int) -> int:
    return sum(math.comb(n_features, r) for r in range(1, max_combo_size + 1))

# Runs the Augmented Dickey-Fuller (ADF) test to assess whether a time series is stationary.
def check_stationarity_adf(series, alpha=0.05, regression='c', autolag='AIC', verbose=True):
    """
    - Drops NaNs and casts series to float.
    - Performs ADF test with chosen regression and lag selection method.
    - Reports test statistic, p-value, critical values, and decision rule.

    Parameters:
        series (pd.Series): Time series to test.
        alpha (float, default=0.05): Significance level for stationarity decision.
        regression (str, default='c'): ADF regression type:
            'c' (constant), 'ct' (constant+trend), 'ctt' (constant+trend+trend²), 'nc' (no constant).
        autolag (str, default='AIC'): Criterion for lag selection ('AIC', 'BIC', 't-stat').
        verbose (bool, default=True): Print results to console.

    Returns:
        dict:
            - adf_stat (float): Test statistic.
            - p_value (float): P-value for unit root.
            - used_lag (int): Number of lags used.
            - nobs (int): Number of observations.
            - critical_values (dict): Critical values for different levels.
            - is_stationary (bool): True if p < alpha (reject unit root).
    """
    s = series.dropna().astype(float)
    n = len(s)
    if n < 10:
        raise ValueError("Series too short for ADF test (need more observations).")

    result = adfuller(s.values, regression=regression, autolag=autolag)
    adf_stat = result[0]
    p_value = result[1]
    usedlag = result[2]
    nobs = result[3]
    crit_values = result[4]

    is_stationary = (p_value < alpha)  # simpler primary decision
    # alternative check: adf_stat < crit_values['5%']

    if verbose:
        print(f"ADF statistic     = {adf_stat:.6f}")
        print(f"p-value           = {p_value:.6f}")
        print(f"used lag          = {usedlag}")
        print(f"n obs (used)      = {nobs}")
        print("critical values   =")
        for k, v in crit_values.items():
            print(f"    {k} : {v:.6f}")
        if is_stationary:
            print("=> ADF: p < {:.2f} -> reject unit root; series is likely STATIONARY.".format(alpha))
        else:
            print("=> ADF: p >= {:.2f} -> cannot reject unit root; series is likely NON-STATIONARY.".format(alpha))

    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'used_lag': usedlag,
        'nobs': nobs,
        'critical_values': crit_values,
        'is_stationary': is_stationary
    }

# Computes Cook's Distance to assess outlier/influential observations in regression.
def cooks_distance(dataframe: pd.DataFrame, fit_model):
    """
    - Uses fitted model's influence measures.
    - Calculates Cook’s D for each observation.
    - Reports top 20 values, threshold (4/n), and counts above thresholds.

    Parameters:
        dataframe (pd.DataFrame): Original data (to get index alignment).
        fit_model (statsmodels fitted model): Fitted OLS or GLM with `.get_influence()` available.

    Returns:
        tuple:
            - cooks (pd.Series): Cook’s D values for all rows (sorted descending).
            - flagged (pd.Series): Observations exceeding 4/n threshold.
    """

    # Influence measures
    influence = fit_model.get_influence()
    cooks_d, cooks_p = influence.cooks_distance

    # Put into DataFrame for inspection
    cooks = pd.Series(cooks_d, index=dataframe.index, name='cooks_d').sort_values(ascending=False)
    top = cooks.head(20)
    print("Top 20 Cook's D values (descending):")
    print(top)

    # thresholds
    n = len(dataframe)
    threshold_4n = 4.0 / n
    print(f"\nThreshold: 4/n = {threshold_4n:.6g}")
    print("Count above 4/n:", (cooks > threshold_4n).sum())
    print("Count above 1.0 :", (cooks > 1.0).sum())

    # return flagged indices
    return cooks, cooks[cooks > threshold_4n]

# (for regression only because of OLS) performs regular i.i.d (independent and identically distributed) bootstrap if block_len = None, otherwise performs moving block bootstrap (for time series) with window overlapping
# returns a dataframe with predictors as index, coef_mean, coed_std, relative_std (normalized by the respective mean), sign_consistency ([0;1]), ci_low, ci_high (confidence interval boundaries)
def bootstrap_coefficients(
    dataframe: pd.DataFrame,     # cleaned DataFrame (train), rows in chronological order
    target: str,                 # target column name (string)
    predictors: np.ndarray,      # list of predictor column names (strings) - same columns used in model
    n_boot: int = 500,           # number of bootstrap samples
    block_len: int = 24,         # 'None' will run -> i.i.d. bootstrap (plain), otherwise MBB block length in rows (e.g., 24 for hourly daily)
    random_state: int = 42
):
    """
    Estimates stability of regression coefficients via bootstrap (i.i.d. or moving-block).

    - If block_len=None: i.i.d. bootstrap (sample rows).
    - If block_len > 0: Moving-block bootstrap (sample blocks of consecutive rows).
    - Fits OLS on each resample.
    - Collects coefficient estimates.
    - Reports mean, std, relative std, sign consistency, and 95% CI.

    Parameters:
        dataframe (pd.DataFrame): Cleaned training data (chronological order).
        target (str): Target variable name.
        predictors (np.ndarray | list[str]): Predictor variable names.
        n_boot (int, default=500): Number of bootstrap replicates.
        block_len (int, default=24): Block length for moving-block bootstrap.
                                     Use None for i.i.d. resampling.
        random_state (int, default=42): RNG seed.

    Returns:
        pd.DataFrame (index=features):
            - coef_mean: Mean bootstrapped coefficient.
            - coef_std: Standard deviation.
            - relative_std: Std normalized by mean magnitude.
            - sign_consistency: Proportion of bootstraps with same sign.
            - ci_low, ci_high: 95% percentile confidence intervals.
    """
    predictors = [c for c in dataframe.columns if c != target]
    
    rng = np.random.RandomState(random_state)
    n = len(dataframe)
    cols = ['const'] + list(predictors)
    collected = []

    if n < 30:
        raise ValueError("Data too small for meaningful bootstrap. Need more rows.")
    if block_len is not None and block_len <= 0:
        block_len = None

    # Build overlapping blocks for moving-block bootstrap if requested
    blocks = None
    n_blocks_needed = None
    if block_len is not None:
        if block_len >= n:
            # fallback to i.i.d. if block_len too large
            block_len = None
        else:
            blocks = []
            for start in range(0, n - block_len + 1):
                blocks.append(dataframe.iloc[start:start+block_len].reset_index(drop=True))
            n_blocks_needed = int(np.ceil(n / block_len))

    for i in range(n_boot):
        if block_len is None:
            # i.i.d. bootstrap: sample indices with replacement
            idx = rng.choice(n, size=n, replace=True)
            sample = dataframe.iloc[idx].reset_index(drop=True)
        else:
            # moving-block bootstrap: sample blocks with replacement and concat
            chosen = [blocks[rng.randint(0, len(blocks))] for _ in range(n_blocks_needed)] # select the built block 'n_blocks_needed' times 
            sample = pd.concat(chosen, ignore_index=True).iloc[:n].reset_index(drop=True) # concat the current selected set of blocks into a dataframe

        # drop rows with NaNs for safety (rare if 'dataframe' cleaned)
        sample = sample[[target] + list(predictors)].dropna()
        if sample.shape[0] < max(30, len(predictors) + 5):
            # if resample leaves too few rows, skip
            continue

        try:
            Xs = sm.add_constant(sample[predictors])
            ys = sample[target]
            model = sm.OLS(ys, Xs).fit()
            params_series = model.params
            params_dict = params_series.to_dict()
            params_vec = np.array([params_dict.get(c, 0.0) for c in cols], dtype=float)
            collected.append(params_vec)
        except Exception as e:
            print(e)
            continue

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{n_boot} bootstraps completed...")

    if len(collected) == 0:
        raise RuntimeError("All bootstrap fits failed — check data / predictors.")

    arr = np.vstack(collected)   # shape (n_effective_boot, n_features)
    mean = arr.mean(axis=0)
    sd   = arr.std(axis=0, ddof=1)

    # relative std (std normalized by the feature's mean)
    eps = 1e-12
    denom = np.where(np.abs(mean) < eps, eps, np.abs(mean)) # if denominator is < 1e-12, we set it to eps=1e-12 to avoid 0 division
    rel_std = sd / denom

    # how often the sign changes from initial means
    mean_sign = np.sign(mean)
    sign_consistency = (np.sign(arr) == mean_sign).mean(axis=0)

    # 95% percentile boundaries
    ci_low = np.percentile(arr, 2.5, axis=0)
    ci_high = np.percentile(arr, 97.5, axis=0)

    out = pd.DataFrame({
        'feature': cols,
        'coef_mean': mean,
        'coef_std': sd,
        'relative_std': rel_std,
        'sign_consistency': sign_consistency,
        'ci_low': ci_low,
        'ci_high': ci_high
    })

    out = out.set_index('feature')
    return out

# Automated hyperparameter search for SARIMAX (seasonal ARIMA without exogenous regressors).
def sarimax_search(y: pd.Series, m: int=24,
                   max_p: int=3, max_d: int=2, max_q: int=3,
                   max_P: int=2, max_D: int=2, max_Q: int=2,
                   mode=Literal["grid", "stepwise"],  # "grid" (only unique combinations but takes longer to fit; used with smaller ranges) if smaller ranges or "stepwise" (tracks improvements but repeats combinations) if mbigger ranges
                   verbose: bool=False):
    """
    - Fits SARIMAX models across ranges of (p,d,q) × (P,D,Q).
    - Supports:
        * Grid search: exhaustive over all combinations.
        * Stepwise search: local hill-climbing (tries neighbors iteratively).
    - Selects best model based on lowest AIC.

    Parameters:
        y (pd.Series): Target time series (must be stationary or differenced).
        m (int, default=24): Seasonal period (e.g., 24 for daily seasonality in hourly data).
        max_p, max_d, max_q (int): Max AR, differencing, MA orders.
        max_P, max_D, max_Q (int): Max seasonal AR, differencing, MA orders.
        mode (str, default="grid"): "grid" (exhaustive) or "stepwise" (faster).
        verbose (bool, default=False): Print candidate AIC values.

    Returns:
        tuple:
            - best_res (SARIMAXResults): Best fitted SARIMAX model.
            - best_order (tuple): Best (p,d,q).
            - best_seasonal (tuple): Best (P,D,Q).
            - best_aic (float): Best AIC score.
    """

    def fit_sarimax(y, order, seasonal_order, m):
        """Helper: try to fit SARIMAX safely, return result or None if fails."""
        try:
            model = sm.tsa.SARIMAX(
                endog=y,
                exog=None, # no need because we only need it in ARIMA mode (to model the seasonal dynamics of the target itself)
                order=order,
                seasonal_order=seasonal_order + (m,),  # seasonal order needs period
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            return res
        except:
            return None

    # Helper: total candidate count for grid
    def total_candidates():
        return ((max_p + 1) * (max_d + 1) * (max_q + 1) *
                (max_P + 1) * (max_D + 1) * (max_Q + 1))

    print(f"Search mode = {mode}")
    if mode == "grid":
        print(f"Total models to fit: {total_candidates()}")
    else:
        print("Worst-case number of models unknown (stepwise may revisit models).")
    
    best_res, best_order, best_seasonal, best_aic = None, None, None, np.inf

    # === GRID SEARCH ===
    if mode == "grid":
        all_orders = [(p, d, q) for p in range(max_p + 1)
                                for d in range(max_d + 1)
                                for q in range(max_q + 1)]
        all_seasonals = [(P, D, Q) for P in range(max_P + 1)
                                     for D in range(max_D + 1)
                                     for Q in range(max_Q + 1)]

        with tqdm(total=len(all_orders) * len(all_seasonals)) as pbar:
            for order in all_orders:
                for seasonal in all_seasonals:
                    res = fit_sarimax(y, order, seasonal, m)
                    pbar.update(1)

                    if res is not None:
                        if verbose:
                            print(f"Trying order={order}, seasonal={seasonal}, m={m}, AIC={res.aic:.2f}")
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res, best_order, best_seasonal = res, order, seasonal

    # === STEPWISE SEARCH ===
    elif mode == "stepwise":
        current_order = (0, 0, 0)
        current_seasonal = (0, 0, 0)
        best_res = fit_sarimax(y, current_order, current_seasonal, m)
        if best_res is None:
            raise ValueError("Baseline SARIMAX could not be fit")

        best_aic = best_res.aic
        best_order, best_seasonal = current_order, current_seasonal
        improved = True

        with tqdm(total=0, position=0, leave=True, dynamic_ncols=True) as pbar:
            while improved:
                improved = False
                neighbors = []

                # Generate neighbors for (p,d,q)
                for delta in [(1,0,0), (-1,0,0),
                              (0,1,0), (0,-1,0),
                              (0,0,1), (0,0,-1)]:
                    cand = tuple(np.clip(np.array(current_order) + np.array(delta),
                                         [0,0,0], [max_p, max_d, max_q]))
                    neighbors.append((cand, current_seasonal))

                # Generate neighbors for (P,D,Q)
                for delta in [(1,0,0), (-1,0,0),
                              (0,1,0), (0,-1,0),
                              (0,0,1), (0,0,-1)]:
                    cand = tuple(np.clip(np.array(current_seasonal) + np.array(delta),
                                         [0,0,0], [max_P, max_D, max_Q]))
                    neighbors.append((current_order, cand))

                # Try all neighbors
                for cand_order, cand_seasonal in neighbors:
                    res = fit_sarimax(y, cand_order, cand_seasonal, m)
                    if res is not None:
                        if verbose:
                            print(f"Trying order={cand_order}, seasonal={cand_seasonal}, m={m}, AIC={res.aic:.2f}")
                        pbar.update(1)

                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res, best_order, best_seasonal = res, cand_order, cand_seasonal
                            current_order, current_seasonal = cand_order, cand_seasonal
                            improved = True

    else:
        raise ValueError("mode must be 'grid' or 'stepwise'")

    return best_res, best_order, best_seasonal, best_aic


# Global holders for the last tuned model and collected results
last_tuned_model = np.nan
df_results = pd.DataFrame()

def tune_model(
    model_class,
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: dict,
    patience: int,
    prev_best_score: float = -np.inf,
    plot_progress: bool = True
) -> pd.DataFrame:

    global last_tuned_model
    global df_results

    # Separate tunable vs. static parameters
    tunable_params = {k: v for k, v in params.items() if isinstance(v, (list, np.ndarray))}
    static_params = {k: v for k, v in params.items() if not isinstance(v, (list, np.ndarray))}

    if len(tunable_params) == 0:
        print("You must specify at least one tunable parameter as a list or ndarray.")
        return

    # Prepare result dataframe
    df_results = pd.DataFrame(columns=list(tunable_params.keys()) + ['train_MAE', 'train_RMSE', 'train_R2', 'test_MAE', 'test_MSE', 'test_R2', 'Time spent (min)'])

    best_rmse = np.inf
    patience_counter = 0
    results_list = []  # <-- collect dicts

    # Create all combinations of tunable parameters
    keys, values = zip(*tunable_params.items())
    param_grid = list(product(*values))

    total_combinations = prod(len(v) for v in tunable_params.values())
    print(f"Total parameter combinations: {total_combinations}")


    for idx, param_combination in enumerate(param_grid):
        current_params = dict(zip(keys, param_combination))
        full_params = {**static_params, **current_params}

        print(f"Processing ({idx+1}): {current_params}")
        last_tuned_model = model_class(**full_params)

        # Time the CV process
        start_time = time.perf_counter()
        cv_results = cross_validate(
            last_tuned_model,
            x_train,
            y_train,
            cv=TimeSeriesSplit(n_splits=5),  # time-series aware split
            scoring=scoring,
            return_train_score=True
        )
        end_time = time.perf_counter()

        # Extract mean metrics
        train_mean_cv_mae = -cv_results['train_MAE'].mean()
        train_mean_cv_rmse = -cv_results['train_RMSE'].mean()
        train_mean_cv_r2 = cv_results['train_R2'].mean()

        test_mean_cv_mae = -cv_results['test_MAE'].mean()
        test_mean_cv_rmse = -cv_results['test_RMSE'].mean()
        test_mean_cv_r2 = cv_results['test_R2'].mean()

        # Store results
        new_row = {
            **current_params,
            'train_MAE': train_mean_cv_mae,
            'train_RMSE': train_mean_cv_rmse,
            'train_R2': train_mean_cv_r2,
            'test_MAE': test_mean_cv_mae,
            'test_RMSE': test_mean_cv_rmse,
            'test_R2': test_mean_cv_r2,
            'Time spent (min)': (end_time - start_time) / 60
        }

        results_list.append(new_row)

        # Save best model (based on RMSE)
        if test_mean_cv_rmse < prev_best_score:
            print(f"🟢 Saved best score: {test_mean_cv_rmse:.4f}")
            prev_best_score = test_mean_cv_rmse
        elif test_mean_cv_rmse == prev_best_score:
            print(f"🟡 Same best score: {test_mean_cv_rmse:.4f}")

        # Early stopping logic
        if test_mean_cv_rmse < best_rmse:
            best_rmse = test_mean_cv_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience != -1 and patience_counter >= patience and idx != len(param_grid) - 1:
                print("Early stopping triggered.")
                break

    df_results = pd.DataFrame(results_list)
    df_results['abs_test_train_RMSE_diff'] = abs(df_results['test_RMSE'] - df_results['train_RMSE'])

    if plot_progress:
        # If train RMSE keeps going down but validation RMSE plateaus or goes up, you see overfitting.
        # If both stay high and close, you see underfitting.
        # If they’re close and low, you’re well-regularized.
        fig, ax = plt.subplots(figsize=(20, 8))

        ax.plot(df_results["train_RMSE"], marker="o", label="Train RMSE")
        ax.plot(df_results["test_RMSE"], marker="x", label="Validation RMSE")
        ax.set_ylabel("RMSE")
        ax.set_title("Train vs Validation RMSE")
        ax.legend()
        ax.grid(True)

        plt.show()
    
    return df_results
