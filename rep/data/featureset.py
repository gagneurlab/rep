from typing import Dict, List, Union

import pyspark
import pyspark.sql.types as t
import pyspark.sql.functions as f

import rep.spark_functions as sf

import logging
log = logging.getLogger(__name__)


__all__ = [
    "transform_featureset",
    "join_featuresets",
]

def transform_featureset(
    fset_df: pyspark.sql.dataframe.DataFrame,
    fset_name: str,
    variables: Dict[str, List[str]],
    index_cols: List[str]
):
    """
    Extracts selected features from some Spark dataframe and flattens them.

    :param fset_df: Some PySpark dataframe
    :param fset_name: Name of the resulting DataFrame
    :param variables: (Nested) list/dictionary of columns which will be used as features.
        All feature columns will be flattened and renamed to "feature.{fset_name}@{alias}"
    :param index_cols: List of index columns which should be kept
    :returns: Spark dataframe
    """
    # create column selection from variables
    feature_columns = list(sf.select_nested_fields(variables))
    # rename feature columns: add prefix
    feature_columns = [col.alias(f"feature.{fset_name}@{alias}") for alias, col in feature_columns]

    # keep only relevant columns from dataframe
    fset_df = fset_df.select(
        # index
        *[c for c in index_cols if c in fset_df.columns],
        # features
        *feature_columns
    )

    return fset_df

def join_featuresets(
    dataframes: Dict[str, pyspark.sql.dataframe.DataFrame],
    variables: Dict[str, Union[List, Dict, str]],
    index_cols: List[str],
    fill_values: Dict[str, object]=None,
    join="outer",
    initial_df=None,
):
    """
    Opens multiple parquet dataset, extracts selected features and joins them to a single dataframe.
    Returns a Spark dataframe.

    :param paths: Dictionary of {fset_name: path} pairs to some parquet datasets
    :param variables: Dictionary of (Nested) list/dictionary of columns which will be used as features.
        All feature columns will be flattened and renamed to "feature.{fset_name}@{alias}"
    :param fill_values: Dictionary of column names and fill values to fill missing values
    :param index_cols: List of index columns on which to join
    :param join: Type of the join.
    :param initial_df: Spark dataframe to join on all loaded feature sets.
        Especially useful for left joins.
    :returns: Spark dataframe
    """
    full_df = initial_df
    for (fset_name, fset_df) in dataframes.items():
        fset_variables = variables[fset_name]

        fset_df = transform_featureset(
            fset_df=fset_df,
            fset_name=fset_name,
            variables=fset_variables,
            index_cols=index_cols,
        )

        if full_df is None:
            full_df = fset_df
            continue

        # determine join columns
        join_columns = [c for c in index_cols if c in fset_df.columns and c in full_df.columns]

        # join all dataframes
        full_df = full_df.join(
            fset_df,
            on=join_columns if len(join_columns) > 0 else None,
            how=join
        )

    if fill_values is not None:
        full_df = full_df.fillna(fill_values)

    return full_df

