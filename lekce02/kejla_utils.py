import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats # kvuli z_scores
from sklearn.linear_model import LinearRegression # kvuli doplneni 
from sklearn.preprocessing import StandardScaler


def null_columns(dataframe: pd.core.frame.DataFrame, relative=False):
    """Eats a dataframe and spits out a dataframe with the nullcount for each column.
    the column with the nullcounts is named 'null_counts'
    the column with the names of columns from the original datafrate is named 'col_names'
    When relative=True is set then it will return the relative nullcount instead
    """

    # get rowcount of original DF:
    length = len(dataframe)
    # get nullcounts per column and turn it into a DF and then rename its columns
    counts_DF = dataframe.isna().sum().reset_index().rename(columns={0: "null_counts", "index": "col_names"})

    # return rel or abs values
    if relative == False:
        return counts_DF
    else:
        counts_DF["null_counts"] /= length
        return counts_DF


def null_columns_only_null(dataframe: pd.core.frame.DataFrame, relative=False):
    """Eats a dataframe and spits out a dataframe with only the columns
    that contain null values. Each column_name with the nullcount for each column
    the column with the nullcounts is named 'null_counts'
    the column with the names of columns from the original datafrate is named 'col_names'"""
    null_columns_DF = null_columns(dataframe, relative=relative)
    return null_columns_DF[null_columns_DF["null_counts"] > 0]


def null_count(dataframe: pd.core.frame.DataFrame, only_null=False, relative=False):
    """Eats a DF and spits out a DF with column names from the original df as rows.
    Each column name with the nullcount for said column.
    The column with the nullcounts is named 'null_counts' in the output DF.
    The column with the names of columns from the original datafrate is named 'col_names
    only_null=True returns only col_names of columns that contain at least one null value
    """

    if only_null == False:
        return null_columns(dataframe, relative=relative)
    else:
        return null_columns_only_null(dataframe, relative=relative)


def unique_values_count(dataframe: pd.core.frame.DataFrame, relative=True):
    """Eats a DF and spits out another DF with the ratio of unique to total values in each column of the original DF.
    When relative=False is set, then it spits out the same only with unique value counts instead of ratios.
    The output DF has column names of the original DF in the column named 'colnames' and the values in the column
    named 'uniqueness'
    """
    # reset index ze series udela DF a pak uz to jenom chceme prejmenovat metodou .rename()
    uniqueness_counts_DF = dataframe.nunique().reset_index().rename(columns={"index": "col_names", 0: "uniqueness"})

    length = len(dataframe)

    # create a copy of df with counts
    temp_DF = uniqueness_counts_DF.copy()

    # scale down the uniqeness by length
    temp_DF["uniqueness"] = temp_DF["uniqueness"] / length
    uniqueness_ratios_DF = temp_DF

    if relative == False:
        return uniqueness_counts_DF
    else:
        return uniqueness_ratios_DF


def get_nullness_and_uniqueness(dataframe: pd.core.frame.DataFrame, only_null=False, relative=True) -> pd.core.frame.DataFrame:

    # this is basically a left join on col_names:
    return pd.merge(null_count(dataframe, relative=relative, only_null=only_null),
                    unique_values_count(dataframe, relative=relative), on="col_names", how="left")
    
def plot_nullness_and_uniqueness(df, only_null=True, relative = True, FIG_SIZE = (10,8), annot=True):
    data = get_nullness_and_uniqueness(df, only_null=only_null, relative=relative)

    #misto umeleho indexu tam dat nas sloupec col_names
    data.index=data["col_names"]

    #smazat sloupec
    data.drop(columns="col_names", inplace=True)
    data
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(data, cmap='coolwarm', annot=annot)
    plt.show()


def get_numeric_colnames(df: pd.core.frame.DataFrame) -> list:
    """returns a list of column names that contain numeric values (np.float64 or np.int64)"""
    return [col_name for col_name in df.select_dtypes(include=np.number)]


def crop_non_numeric_columns(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """returns only the numeric columns as a dataframe"""
    numeric_cols_DF = pd.DataFrame()
    for col_name in get_numeric_colnames(df):
        numeric_cols_DF[col_name] = df[col_name]
    return numeric_cols_DF


def get_cormatrix(df: pd.core.frame.DataFrame):
    data = crop_non_numeric_columns(df)
    return data.corr()


def plot_corhmap(df: pd.core.frame.DataFrame, FIG_SIZE = (10,8), annot=True):
    """plots a correlation heatmap for df numeric values"""
    corm_DF = get_cormatrix(df)
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(corm_DF, cmap="coolwarm", annot=annot)
    plt.show()


def isnull_replace(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    data_AVG_imput = df.copy()
    for col_name in get_numeric_colnames(data_AVG_imput):
        # print(col_name)
        data_AVG_imput[col_name] = data_AVG_imput[col_name].fillna(data_AVG_imput[col_name].mean())
    return data_AVG_imput


def get_most_cor_colnames_to_target(df: pd.core.frame.DataFrame, input_col_N: int,
                                    target_colname: str) -> list[str]:
    input_colnames = []
    corr_col = get_cormatrix(df)[target_colname]

    # drop target:
    other_corr_col = corr_col.drop(target_colname)

    # absolute values:
    for i in other_corr_col.index:
        other_corr_col[i] = abs(other_corr_col[i])
    # collect highest correlated
    for col_ind in range(input_col_N):
        found_colname = other_corr_col[other_corr_col == other_corr_col.max()].index[0]
        input_colnames.append(found_colname)
        other_corr_col.drop(found_colname, inplace=True)
    return input_colnames


def fill_nulls_lin_reg(df: pd.core.frame.DataFrame, target_colname: str, input_col_N: int) -> pd.core.frame.DataFrame:

    input_col_names = get_most_cor_colnames_to_target(df, input_col_N, target_colname)

    nulls_filled_lin_reg = df.copy()

    data_to_train_model_DF = df.dropna(subset=target_colname)

    # nafitovat lin_reg:
    model = LinearRegression().fit(data_to_train_model_DF[input_col_names], data_to_train_model_DF[target_colname])

    rows_with_nulls_in_target = df[df[target_colname].isnull()]

    # model.predict(...) vraci hodnoty na zaklade treninku:
    ##deprecated:
    ##nulls_filled_lin_reg[target_colname][nulls_filled_lin_reg[target_colname].isnull()] = model.predict(rows_with_nulls_in_target[input_col_names])
    nulls_filled_lin_reg.loc[nulls_filled_lin_reg[target_colname].isnull(), target_colname] = model.predict(
        rows_with_nulls_in_target[input_col_names])

    return nulls_filled_lin_reg


def get_outliers(df, colname: str, threshold: float) -> pd.core.frame.DataFrame:
    data = df.copy()
    data['zscore'] = stats.zscore(data[colname])
    return data[abs(data['zscore']) > threshold]


def one_hot_encode(df, colnames: list[str]) -> pd.core.frame.DataFrame:
    return pd.get_dummies(df, columns=colnames)


def plot_z_scores_scatter(df, x_colname="x_column", y_colname="y_column", N_categories = 4 ,figsize = (10,8)):
    colors_universe = ["blue", "green", "red", "cyan", "magenta", "yellow", "black",
          "orange", "purple", "brown", "pink", "gray", "olive", "teal", "navy"]
    colors = colors_universe[:N_categories]
    plt.figure(figsize=figsize)
    for i, color in enumerate(colors):
        #print(f"{i} {color}")
        curr_outliers_DF = get_outliers(df, x_colname, i)

        plt.scatter(curr_outliers_DF[x_colname], curr_outliers_DF[y_colname], color=color, label = f"z_score > {i}")

    plt.legend()
    plt.show()