from typing import Dict, List, Union

import polars as pl
import polars.datatypes as t

from .reshape import select_nested_fields

import logging
log = logging.getLogger(__name__)


__all__ = [
    "transform_featureset",
    "join_featuresets",
]


def transform_featureset(
    fset_df: Union[pl.DataFrame, pl.LazyFrame],
    fset_name: str,
    variables: Dict[str, List[str]],
    index_cols: List[str]
):
    """
    Extracts selected features from some Spark dataframe and flattens them.

    :param fset_df: Some Polars dataframe
    :param fset_name: Name of the resulting DataFrame
    :param variables: (Nested) list/dictionary of columns which will be used as features.
        All feature columns will be flattened and renamed to "feature.{fset_name}@{alias}"
    :param index_cols: List of index columns which should be kept
    :returns: Spark dataframe
    """
    # create column selection from variables
    feature_columns = list(select_nested_fields(variables))
    # rename feature columns: add prefix
    feature_columns = [col.alias(f"feature.{fset_name}@{alias}") for alias, col in feature_columns]

    # keep only relevant columns from dataframe
    fset_df = fset_df.select([
        # index
        *[c for c in index_cols if c in fset_df.columns],
        # features
        *feature_columns
    ])

    return fset_df


def join_featuresets(
    dataframes: Dict[str, Union[pl.DataFrame, pl.LazyFrame]],
    variables: Dict[str, Union[List, Dict, str]],
    index_cols: List[str],
    fill_values: Dict[str, object]=None,
    join="outer",
    initial_df: Union[pl.DataFrame, pl.LazyFrame]=None,
):
    """
    Opens multiple parquet dataset, extracts selected features and joins them to a single dataframe.
    Returns a Polars dataframe.

    :param paths: Dictionary of {fset_name: path} pairs to some parquet datasets
    :param variables: Dictionary of (Nested) list/dictionary of columns which will be used as features.
        All feature columns will be flattened and renamed to "feature.{fset_name}@{alias}"
    :param fill_values: Dictionary of column names and fill values to fill missing values
    :param index_cols: List of index columns on which to join
    :param join: Type of the join.
    :param initial_df: dataframe to join on all loaded feature sets.
        Especially useful for left joins.
    :returns: Polars dataframe
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
        fill_exprs = []
        for col, fval in fill_values.items():
            expr = pl.col(col).fill_null(fval)
            if full_df.schema[col] in t.FLOAT_DTYPES:
                expr = expr.fill_nan(fval)
            expr = expr.alias(col)
            
            fill_exprs.append(expr)
        
        full_df = full_df.with_columns(fill_exprs)


    return full_df

