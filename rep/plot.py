import numpy.random as nr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns

from sklearn import metrics


def roc_curve(y_true, y_pred, label, formatting='%s (auROC = %0.2f%%)'):
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_true, y_pred)
    # Now, plot the computed values
    plt.plot(fpr, tpr, label=formatting % (label, 100 * auc))


def roc_plot(y_trues, y_preds: list, labels=None, add_random_shuffle=False, legend_pos="inside"):
    plt.figure()

    if not isinstance(y_preds, list):
        raise ValueError("y_preds is not a list")

    if not isinstance(y_trues, list) and not np.ndim(y_trues) > 1:
        y_trues = [y_trues] * len(y_preds)

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

    formatting = '%s (auROC = %0.2f%%)' if legend_pos == "inside" else '%s\n(auROC = %0.2f%%)'

    # Below for loop iterates through your models list
    for l, y_true, y_pred in zip(labels, y_true, np.asarray(y_preds).T):
        roc_curve(y_true, y_pred, label=l, formatting=formatting)

    if add_random_shuffle:
        roc_curve(np.random.choice(y_true, len(y_true)), label="random shuffle", formatting=formatting)

    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1], 'r--')
    # plt.grid()
    plt.xlabel('False Positive Rate FP/TN+FP')
    plt.ylabel('True Positive Rate TP/TP+FP')
    plt.title('Receiver Operating Characteristic')

    if legend_pos == "inside":
        plt.legend(loc=4)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plt.tight_layout()

    return plt.gcf()


def precision_recall_curve(y_true, y_pred, label, formatting='%s (auc = %0.2f%%)'):
    # Compute precision and recall
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    # Calculate Area under the curve to display on the plot
    auc = metrics.auc(recall, precision)
    # Now, plot the computed values
    plt.plot(recall, precision, label=formatting % (label, 100 * auc))


def precision_recall_plot(y_trues, y_preds, labels=None, add_random_shuffle=True, legend_pos="inside"):
    plt.figure()

    if not isinstance(y_preds, list):
        raise ValueError("y_preds is not a list")

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

    formatting = '%s (auc = %0.2f%%)' if legend_pos == "inside" else '%s\n(auc = %0.2f%%)'

    # Below for loop iterates through your models list
    for l, y_true, y_pred in zip(labels, y_trues, y_preds):
        precision_recall_curve(y_true, y_pred, label=l, formatting=formatting)

    if add_random_shuffle:
        precision_recall_curve(y_true, np.random.choice(y_true, len(y_true)), label="random shuffle",
                               formatting=formatting)

    # Custom settings for the plot
    # plt.grid()
    plt.xlabel("recall TP/(TP+FN)")
    plt.ylabel("precision TP/(TP+FP)")
    plt.title("Precision vs. Recall")

    if legend_pos == "inside":
        plt.legend(loc=4)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plt.tight_layout()

    return plt.gcf()


def tp_at_k(observed, score):
    df = pd.DataFrame(dict(observed=observed, score=score))
    df = df.sort_values(by="score", ascending=False).reset_index()
    df["n_true"] = df["observed"].cumsum()
    df["k"] = df.index

    return df


def tp_at_k_curve(y_true, y_pred, label, formatting='%s (auc = %0.2f%%)', y_true_sum=None):
    if not y_true_sum:
        y_true_sum = np.asarray(np.sum(y_true))

    # Compute precision at k
    df = tp_at_k(y_true, y_pred)
    # Calculate Area under the curve to display on the plot
    auc = metrics.auc(df["k"] / len(y_true), df["n_true"] / y_true_sum)
    # Now, plot the computed values
    plt.plot(df["k"], df["n_true"], label=formatting % (label, 100 * auc))


def tp_at_k_plot(y_trues, y_preds, labels=None, add_random_uniform=False, legend_pos="inside"):
    plt.figure()

    if not isinstance(y_preds, list):
        raise ValueError("y_preds is not a list")

    # calculate sum only if there is only a single y_true array
    y_true_sum = None
    if not isinstance(y_trues, list) and not np.ndim(y_trues) > 1:
        y_true_sum = np.asarray(np.sum(y_trues))
        y_trues = [y_trues] * len(y_preds)

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

    formatting = '%s (auc = %0.2f%%)' if legend_pos == "inside" else '%s\n(auc = %0.2f%%)'

    # Below for loop iterates through your models list
    for l, y_true, y_pred in zip(labels, y_trues, np.asarray(y_preds).T):
        tp_at_k_curve(y_true, y_pred, label=l, formatting=formatting, y_true_sum=y_true_sum)

    if add_random_uniform:
        tp_at_k_curve(y_true, np.random.uniform(size=len(y_true)), label="random uniform", formatting=formatting,
                      y_true_sum=y_true_sum)

    # Custom settings for the plot
    plt.plot([0, len(y_true)], [0, y_true_sum], 'r--')
    plt.xlabel("k (rank of score)")
    plt.ylabel("number of true positives")
    plt.title("True Positives at k")

    if legend_pos == "inside":
        plt.legend(loc=4)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plt.tight_layout()

    return plt.gcf()


def density_scatter(
        x,
        y,
        data: pd.DataFrame = None,
        xlab="x",
        ylab="y",
        xlim=None,
        ylim=None,
        sort=True,
        bins=1000,
        marker_size=1,
        marker_linewidth=0,
        marker_colornorm=None,
        rasterized=True,
        normalize_density=True,
        kde=False,
        scatter_kwargs=None,
        distplot_kwargs=None,
        jointgrid_kwargs=None,
        **kwargs
):
    """
    Scatter plot colored by 2d histogram.

    Example:
        ```
        g = density_scatter(
            x=m.predicted.values,
            y=m.zscore.values,
            ylab='z-score (difference to mean)\nlung', 
            xlab='prediction score\nlung', 
            jointgrid_kwargs=dict(
                xlim=[-10, 10], # x-axis limits
                ylim=[-10, 10], # x-axis limits
            ),
            scatter_kwargs=dict(
                rasterized=True, # do not use vector graphics for scatter plot
                norm=matplotlib.colors.LogNorm(), # log-scale for point density
                sizes=(1, 1), # point size
            ),
        )
        g.fig.suptitle(model)
        plt.show()
        ```
    """
    if data is not None:
        x = data.loc[:, x].values
        y = data.loc[:, y].values

    from scipy.interpolate import interpn
    xy = pd.DataFrame({xlab: x, ylab: y})
    xy = xy.dropna().reset_index(drop=True)
    # keep data as pandas series for axis labels (!)
    x: pd.Series = xy[xlab]
    y: pd.Series = xy[ylab]
    # calculate 2D histogram
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    # interpolate every point based on the distance to the neighbouring bin
    z: np.ndarray = interpn(
        (
            0.5 * (x_e[1:] + x_e[:-1]),
            0.5 * (y_e[1:] + y_e[:-1])
        ),
        data,
        xy,
        method="splinef2d",
        bounds_error=False,
        fill_value=0
    )

    if normalize_density:
        # normalize minimal value to 1, especially important for logarithmic color scale
        z = z - np.min(z) + 1

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x = x.iloc[idx]
        y = y.iloc[idx]
        z = z[idx]

    # set default scatter_kwargs
    combined_scatter_kwargs = dict(
        s=marker_size,
        linewidth=marker_linewidth,
        rasterized=rasterized,
    )
    if marker_colornorm:
        if (type(marker_colornorm) == str) and marker_colornorm.lower() == "log":
            combined_scatter_kwargs["norm"] = matplotlib.colors.LogNorm()
        else:
            combined_scatter_kwargs["norm"] = marker_colornorm
    # default scatter_kwargs will be overridden by user-defined options
    if scatter_kwargs is not None:
        for k, v in scatter_kwargs.items():
            combined_scatter_kwargs[k] = v

    # set up default jointgrid args
    combined_jointgrid_kwargs = dict()
    if xlim:
        combined_jointgrid_kwargs["xlim"] = xlim
    if ylim:
        combined_jointgrid_kwargs["ylim"] = ylim
    # default jointgrid_kwargs will be overridden by user-defined options
    if jointgrid_kwargs is not None:
        for k, v in jointgrid_kwargs.items():
            combined_jointgrid_kwargs[k] = v

    combined_distplot_kwargs = dict(
        kde=kde  # kde takes a long time to calculate
    )
    # default distplot_kwargs will be overridden by user-defined options
    if distplot_kwargs is not None:
        for k, v in distplot_kwargs.items():
            combined_distplot_kwargs[k] = v

    # create the JointGrid
    g = sns.JointGrid(x=x, y=y, **combined_jointgrid_kwargs)
    g = g.plot_marginals(sns.distplot, **combined_distplot_kwargs)
    # hack to get the correct coordinates set in plt.scatter
    g.x = x
    g.y = y
    g = g.plot_joint(plt.scatter, c=z, **combined_scatter_kwargs)
    # shrink fig so cbar is visible
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    # make new ax object for the cbar
    cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    return g
