import abc
from typing import Callable, Dict, Union, List, Tuple

import copy

import pandas as pd
import pandera as pdt

try:
    from functools import cached_property
except ImportError:
    # python < 3.8
    from backports.cached_property import cached_property

import desmi
from rep.data.desmi import GTExTranscriptProportions


__all__ = [
    "PandasTransformer",
    "LambdaTransformer",
    "Aggregator",
]

def _clear_checks(schema: Union[pdt.DataFrameSchema, pdt.SeriesSchema], inplace=False):
    if not inplace:
        schema = copy.deepcopy(schema)

    # remove schema checks:
    schema.checks.clear()

    # remove column checks when schema describes a DataFrame:
    if isinstance(schema, pdt.DataFrameSchema):
        for col_name, col_schema in schema.columns.items():
            col_schema.checks.clear()

    return schema


class PandasTransformer(metaclass=abc.ABCMeta):
    """
    Base class for Pandas transformers.
    Assumes a stable output schema independent of the input.
    """

    @property
    @abc.abstractmethod
    def schema(self, *args, **kwargs) -> Union[pdt.SeriesSchema, pdt.DataFrameSchema]:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.schema)})"


class LambdaTransformer(PandasTransformer, metaclass=abc.ABCMeta):

    def __init__(self, transformer_fn, *args, **kwargs):
        """
        Annotate some Pandas transformation function with an inferred schema.
        Needs example arguments to run the function once.

        Example:

        .. code-block:: python

            mean_transformer = LambdaTransformer(lambda series: series.mean(), pd.Series(range(4)))

            assert mean_transformer.schema.pandas_dtype == "float64"
            assert mean_transformer(pd.Series(range(10))) == 4.5

        :param transformer_fn: The function that should be annotated
        :param args: Example arguments to run the function
        :param kwargs: Example keyword arguments to run the function
        """
        self.transformer_fn = transformer_fn

        example_data = self.__call__(*args, **kwargs)
        if not isinstance(example_data, pd.DataFrame) or isinstance(example_data, pd.Series):
            # convert to Series by default
            example_data = pd.Series(example_data)

        # now infer schema
        self._schema = pdt.infer_schema(example_data)
        # remove schema checks:
        self._schema = _clear_checks(self._schema)

    @property
    def schema(self) -> pdt.DataFrameSchema:
        return self._schema

    def __call__(self, *args, **kwargs):
        return self.transformer_fn(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.transformer_fn)}, {repr(self.schema)})"


class Aggregator(PandasTransformer):
    def __init__(
            self,
            input_schema: pdt.DataFrameSchema,
            groupby: Union[str, List[str]],
            agg: Dict[str, List[Union[str, Tuple[str, Union[str, Callable, PandasTransformer]]]]],
    ):
        self.input_schema = input_schema
        if isinstance(groupby, str):
            # make sure that groupby is a list
            groupby = [groupby]
        self.groupby = groupby
        self.agg = agg

    @cached_property
    def schema(self) -> pdt.DataFrameSchema:
        new_schema = self.input_schema.reset_index().select_columns(self.groupby).set_index(self.groupby)

        # add columns: {col_name: [(func_name, func)]+}
        new_columns = {}
        for col_name, func_list in self.agg.items():
            col_schema: pdt.Column = self.input_schema.columns[col_name]
            for item in func_list:
                if isinstance(item, str):
                    func_name, func = item, item
                else:
                    try:
                        func_name, func = item
                    except ValueError as e:
                        raise ValueError(f"Invalid argument for aggregating column '{col_name}': '{item}'") from e

                if isinstance(func, PandasTransformer):
                    func_return_type = func.schema.pandas_dtype
                elif isinstance(func, str):
                    func_return_type = pdt.infer_schema(
                        col_schema.example(1)
                            .iloc[:, 0]
                            .agg([func])
                    ).pandas_dtype
                else:
                    func_return_type = LambdaTransformer(func, col_schema.example(1)).schema.pandas_dtype
                new_columns[(col_name, func_name)] = pdt.Column(func_return_type)
        new_schema = new_schema.add_columns(new_columns)

        return new_schema

    def __call__(self, input: pd.DataFrame) -> pd.DataFrame:
        return input.groupby(self.groupby).agg(self.agg)


def test_lambda_transformer():
    mean_transformer = LambdaTransformer(lambda series: series.mean(), pd.Series(range(4)))

    assert mean_transformer.schema.pandas_dtype == "float64"
    assert mean_transformer(pd.Series(range(10))) == 4.5


def test_dataframe_aggregator():
    df = pd.DataFrame(dict(a=["x", "y", "x"], b=[1, 2, 3]))
    df_schema = _clear_checks(pdt.infer_schema(df))

    df_agg_schema = _clear_checks(pdt.infer_schema(
        pd.DataFrame(dict(a=["x", "y", "x"], b=[1, 2, 3])).groupby("a").agg(["min", "max"])
    ))

    agg = Aggregator(df_schema, groupby="a", agg={"b": ["min", "max"]})

    assert agg.schema.to_yaml() == df_agg_schema.to_yaml()
    assert _clear_checks(pdt.infer_schema(agg(df))).to_yaml() == df_agg_schema.to_yaml()

    agg = Aggregator(df_schema, groupby="a", agg={"b": [
        ("min", LambdaTransformer(lambda series: series.min(), df_schema.columns["b"].example(10).iloc[:, 0])),
        ("max", lambda series: series.max()),
    ]})

    assert agg.schema.to_yaml() == df_agg_schema.to_yaml()
    assert _clear_checks(pdt.infer_schema(agg(df))).to_yaml() == df_agg_schema.to_yaml()
