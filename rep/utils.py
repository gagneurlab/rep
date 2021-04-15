from typing import List

import numpy as np
import numpy.typing
import pandas as pd


def str_normalize(
        names: numpy.typing.ArrayLike,
        regex="\s|_|\(|\)|-",
        replace="",
        dtype=pd.StringDtype()
) -> pd.Series:
    names = pd.Series(np.unique(names), dtype=dtype)
    norm_names = names.str.replace(regex, replace, regex=True).str.lower()
    if len(np.unique(norm_names)) != len(norm_names):
        raise ValueError("Could not normalize strings: non-unique mapping after regex!")
    return names.set_axis(norm_names)


def fuzzy_match_names(
        names_a: numpy.typing.ArrayLike,
        names_b: numpy.typing.ArrayLike,
        regex="\s|_|\(|\)|-",
        replace="",
        dtype=pd.StringDtype()
) -> pd.DataFrame:
    names_a = str_normalize(names_a, regex=regex, replace=replace, dtype=dtype)
    names_b = str_normalize(names_b, regex=regex, replace=replace, dtype=dtype)
    matching_df = pd.DataFrame({
        "names_a": names_a,
        "names_b": names_b
    })
    return matching_df


def fuzzy_match_to_reference(
        ref_names: numpy.typing.ArrayLike,
        alt_names: numpy.typing.ArrayLike,
        regex="\s|_|\(|\)|-",
        replace="",
        dtype=pd.StringDtype()
) -> numpy.typing.ArrayLike:
    """
    Matches a sequence of strings to a set of reference names by
    matching to common string "hashes" using the specified regex.
    Requires that there can be found a 1:1 mapping between all unique values in `ref_names` and `alt_names`.

    Example:


    :param ref_names: Sequence of names that should be matched to
    :param alt_names:
    :param dtype:
    :return: Sequence of names with same length as `len(alt_names)`.
        Each element is either an element from `ref_names` or N/A.
    """
    matching_df = fuzzy_match_names(ref_names, alt_names, regex=regex, replace=replace, dtype=dtype)
    return matching_df.set_index("names_b")["names_a"].loc[alt_names].values


def test_fuzzy_match_to_reference():
    ref_names = ["X - T", "X - T", "Y - A"]
    alt_names = pd.Series(["Y___A", "Y___A", "B_Y", "X___T", "X___T"], dtype=pd.StringDtype())
    exp_names = pd.Series(['Y - A', 'Y - A', pd.NA, 'X - T', 'X - T'], dtype=pd.StringDtype())

    assert exp_names.equals(pd.Series(fuzzy_match_to_reference(ref_names, alt_names)))
