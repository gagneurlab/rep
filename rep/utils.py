import numpy as np
import numpy.typing
import pandas as pd
from sklearn import metrics


def str_normalize(
    names: numpy.typing.ArrayLike,
    regex="\s|_|\(|\)|-",
    replace="",
    dtype=pd.StringDtype(),
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
    dtype=pd.StringDtype(),
) -> pd.DataFrame:
    names_a = str_normalize(names_a, regex=regex, replace=replace, dtype=dtype)
    names_b = str_normalize(names_b, regex=regex, replace=replace, dtype=dtype)
    matching_df = pd.DataFrame({"names_a": names_a, "names_b": names_b})
    return matching_df


def fuzzy_match_to_reference(
    ref_names: numpy.typing.ArrayLike,
    alt_names: numpy.typing.ArrayLike,
    regex="\s|_|\(|\)|-",
    replace="",
    dtype=pd.StringDtype(),
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
    matching_df = fuzzy_match_names(
        ref_names, alt_names, regex=regex, replace=replace, dtype=dtype
    )
    return matching_df.set_index("names_b")["names_a"].loc[alt_names].values


def test_fuzzy_match_to_reference():
    ref_names = ["X - T", "X - T", "Y - A"]
    alt_names = pd.Series(
        ["Y___A", "Y___A", "B_Y", "X___T", "X___T"], dtype=pd.StringDtype()
    )
    exp_names = pd.Series(
        ["Y - A", "Y - A", pd.NA, "X - T", "X - T"], dtype=pd.StringDtype()
    )

    assert exp_names.equals(pd.Series(fuzzy_match_to_reference(ref_names, alt_names)))


def _prc_step(precision, recall, sklearn_mode=True):
    if not sklearn_mode:
        # make sure that input is sorted by recall
        idx = np.argsort(recall)
    else:
        # by default, sklearn reports recall sorted descending
        # => just inverting in this case is faster
        idx = slice(None, None, -1)

    prec_step = np.zeros((len(precision) * 2) - 1)
    rec_step = np.zeros((len(recall) * 2) - 1)

    prec_step[np.arange(len(precision)) * 2] = precision[idx]
    rec_step[np.arange(len(recall)) * 2] = recall[idx]

    # resemble 'post' step plot
    prec_step[np.arange(len(recall) - 1) * 2 + 1] = precision[idx][1:]
    rec_step[np.arange(len(recall) - 1) * 2 + 1] = recall[idx][:-1]

    return prec_step, rec_step


def get_prc_step_curve(y_true, y_pred, binary_as_point=True):
    if binary_as_point:
        min_y_pred = np.min(y_pred)
        max_y_pred = np.max(y_pred)

        if np.all(np.isin(y_pred, [min_y_pred, max_y_pred])):
            # y_pred is a binary predictor
            binarized_y_pred = y_pred == max_y_pred

            recall = [metrics.recall_score(y_true, binarized_y_pred)]
            precision = [metrics.precision_score(y_true, binarized_y_pred)]

            return precision, recall

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true=y_true,
        probas_pred=y_pred,
    )
    return _prc_step(precision, recall, sklearn_mode=True)


def get_prc_step_curve_df_single(y_true, y_pred, binary_as_point=True):
    prec_step, rec_step = get_prc_step_curve(
        y_true=y_true, y_pred=y_pred, binary_as_point=binary_as_point
    )
    auc = metrics.average_precision_score(y_true=y_true, y_score=y_pred)

    return pd.DataFrame(
        {
            "recall": rec_step,
            "precision": prec_step,
            "is_binary": len(prec_step) == 1,
            "auc": auc,
        }
    )


def get_prc_step_curve_df(y_trues, y_preds, labels, binary_as_point=True):
    if not isinstance(y_preds, list) and not np.ndim(y_preds) > 1:
        y_preds = [y_preds]

    if not isinstance(y_trues, list) and not np.ndim(y_trues) > 1:
        y_trues = [y_trues] * len(y_preds)

    # infer labels
    if isinstance(y_preds, pd.DataFrame):
        if labels is None:
            labels = y_preds.columns.astype(str)
    else:
        if labels is None:
            labels = np.arange(1, np.shape(y_preds)[1] + 1)
    labels = np.asarray(labels).flatten()

    # convert DataFrame to list
    if isinstance(y_preds, pd.DataFrame):
        y_preds = [v.values for k, v in y_preds.items()]

    dfs = []
    # Below for loop iterates through your models list
    for l, y_true, y_pred in zip(labels, y_trues, y_preds):
        df = get_prc_step_curve_df_single(
            y_true, y_pred, binary_as_point=binary_as_point
        )
        df["model"] = l
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def get_prc_curve_df_single(y_true, y_pred, binary_as_point=True):
    if binary_as_point:
        min_y_pred = np.min(y_pred)
        max_y_pred = np.max(y_pred)

        if np.all(np.isin(y_pred, [min_y_pred, max_y_pred])):
            # y_pred is a binary predictor
            binarized_y_pred = y_pred == max_y_pred

            recall = [metrics.recall_score(y_true, binarized_y_pred)]
            precision = [metrics.precision_score(y_true, binarized_y_pred)]
            thresholds = [(float(max_y_pred) - float(min_y_pred)) / 2]

    auc = metrics.average_precision_score(y_true=y_true, y_score=y_pred)

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true=y_true,
        probas_pred=y_pred,
    )
    recall = recall[: len(thresholds)]
    precision = precision[: len(thresholds)]

    prc_df = (
        pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": thresholds,
                "is_binary": len(thresholds) == 1,
                "auc": auc,
            }
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )

    return prc_df


def get_prc_curve_df(y_trues, y_preds, labels, binary_as_point=True):
    if not isinstance(y_preds, list) and not np.ndim(y_preds) > 1:
        y_preds = [y_preds]

    if not isinstance(y_trues, list) and not np.ndim(y_trues) > 1:
        y_trues = [y_trues] * len(y_preds)

    # infer labels
    if isinstance(y_preds, pd.DataFrame):
        if labels is None:
            labels = y_preds.columns.astype(str)
    else:
        if labels is None:
            labels = np.arange(1, np.shape(y_preds)[1] + 1)
    labels = np.asarray(labels).flatten()

    # convert DataFrame to list
    if isinstance(y_preds, pd.DataFrame):
        y_preds = [v.values for k, v in y_preds.items()]

    dfs = []
    # Below for loop iterates through your models list
    for l, y_true, y_pred in zip(labels, y_trues, y_preds):
        df = get_prc_curve_df_single(y_true, y_pred, binary_as_point=binary_as_point)
        df["model"] = l
        dfs.append(df)
    df = pd.concat(dfs)
    return df
