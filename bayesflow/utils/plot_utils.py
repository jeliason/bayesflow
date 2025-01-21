from typing import Sequence, Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .validators import check_posterior_prior_shapes
from .dict_utils import dicts_to_arrays


def prepare_plot_data(
    targets: Mapping[str, np.ndarray] | np.ndarray,
    references: Mapping[str, np.ndarray] | np.ndarray,
    variable_names: Sequence[str] = None,
    num_col: int = None,
    num_row: int = None,
    figsize: tuple = None,
    stacked: bool = False,
    default_name: str = "var",
) -> Mapping[str, Any]:
    """
    Procedural wrapper that encompasses all preprocessing steps, including shape-checking, parameter name
    generation, layout configuration, figure initialization, and collapsing of axes.

    Parameters
    ----------
    targets           : dict[str, ndarray] or ndarray
        The model-generated predictions or estimates, which can take the following forms:
        - ndarray of shape (num_datasets, num_variables)
            Point estimates for each dataset, where `num_datasets` is the number of datasets
            and `num_variables` is the number of variables per dataset.
        - ndarray of shape (num_datasets, num_draws, num_variables)
            Posterior samples for each dataset, where `num_datasets` is the number of datasets,
            `num_draws` is the number of posterior draws, and `num_variables` is the number of variables.
    references        : dict[str, ndarray] or ndarray, optional (default = None)
        Ground truth values corresponding to the estimates. Must match the structure and dimensionality
        of `estimates` in terms of first and last axis.
    variable_names    : Sequence[str], optional (default = None)
        Optional variable names to act as a filter if dicts provided or actual variable names in case of array args
    num_col           : int
        Number of columns for the visualization layout
    num_row           : int
        Number of rows for the visualization layout
    figsize           : tuple, optional, default: None
        Size of the figure adjusting to the display resolution
    stacked           : bool, optional, default: False
        Whether the plots are stacked horizontally
    default_name      : str, optional (default = "var")
        The default name to use for targets if None provided
    """

    plot_data = dicts_to_arrays(
        targets=targets, references=references, variable_names=variable_names, default_name=default_name
    )
    check_posterior_prior_shapes(plot_data["targets"], plot_data["references"])

    # Configure layout
    num_row, num_col = set_layout(plot_data["num_variables"], num_row, num_col, stacked)

    # Initialize figure
    fig, axes = make_figure(num_row, num_col, figsize=figsize)

    if num_row == 1 and num_col == 1:
        axes = np.array([axes])

    plot_data["fig"] = fig
    plot_data["axes"] = axes
    plot_data["num_row"] = num_row
    plot_data["num_col"] = num_col

    return plot_data


def set_layout(num_total: int, num_row: int = None, num_col: int = None, stacked: bool = False):
    """
    Determine the number of rows and columns in diagnostics visualizations.

    Parameters
    ----------
    num_total     : int
        Total number of parameters
    num_row       : int, default = None
        Number of rows for the visualization layout
    num_col       : int, default = None
        Number of columns for the visualization layout
    stacked     : bool, default = False
        Boolean that determines whether to stack the plot or not.

    Returns
    -------
    num_row       : int
        Number of rows for the visualization layout
    num_col       : int
        Number of columns for the visualization layout
    """
    if stacked:
        num_row, num_col = 1, 1
    else:
        if num_row is None and num_col is None:
            num_row = int(np.ceil(num_total / 6))
            num_col = int(np.ceil(num_total / num_row))
        elif num_row is None and num_col is not None:
            num_row = int(np.ceil(num_total / num_col))
        elif num_row is not None and num_col is None:
            num_col = int(np.ceil(num_total / num_row))

    return num_row, num_col


def make_figure(num_row: int = None, num_col: int = None, figsize: tuple = None):
    """
    Initialize a set of figures

    Parameters
    ----------
    num_row       : int
        Number of rows in a figure
    num_col       : int
        Number of columns in a figure
    figsize       : tuple
        Size of the figure adjusting to the display resolution
        or the user's choice

    Returns
    -------
    f, axes
        Initialized figures
    """
    if num_row == 1 and num_col == 1:
        f, axes = plt.subplots(1, 1, figsize=figsize)
    else:
        if figsize is None:
            figsize = (int(5 * num_col), int(5 * num_row))

        f, axes = plt.subplots(num_row, num_col, figsize=figsize)

    return f, axes


def add_metric(
    ax,
    metric_text: str = None,
    metric_value: float = None,
    position: tuple = (0.1, 0.9),
    metric_fontsize: int = 12,
):
    """TODO: docstring"""
    if metric_text is None or metric_value is None:
        raise ValueError("Metric text and values must be provided to be add this metric.")

    metric_label = f"{metric_text} = {metric_value:.3f}"

    ax.text(
        position[0],
        position[1],
        metric_label,
        ha="left",
        va="center",
        transform=ax.transAxes,
        size=metric_fontsize,
    )


def add_x_labels(
    axes: np.ndarray,
    num_row: int = None,
    num_col: int = None,
    xlabel: Sequence[str] | str = None,
    label_fontsize: int = None,
):
    """TODO: docstring"""
    if num_row == 1:
        bottom_row = axes
    else:
        bottom_row = axes[num_row - 1, :] if num_col > 1 else axes
    for i, ax in enumerate(bottom_row):
        # If labels are in sequence, set them sequentially. Otherwise, one label fits all.
        ax.set_xlabel(xlabel if isinstance(xlabel, str) else xlabel[i], fontsize=label_fontsize)


def add_y_labels(axes: np.ndarray, num_row: int = None, ylabel: Sequence[str] | str = None, label_fontsize: int = None):
    """TODO: docstring"""

    if num_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylabel, fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for i, ax in enumerate(axes[:, 0]):
            # If labels are in sequence, set them sequentially. Otherwise, one label fits all.
            ax.set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[i], fontsize=label_fontsize)


def add_titles(axes: np.ndarray, title: Sequence[str] | str = None, title_fontsize: int = None):
    for i, ax in enumerate(axes.flat):
        ax.set_title(title[i], fontsize=title_fontsize)


def add_titles_and_labels(
    axes: np.ndarray,
    num_row: int = None,
    num_col: int = None,
    title: Sequence[str] | str = None,
    xlabel: Sequence[str] | str = None,
    ylabel: Sequence[str] | str = None,
    title_fontsize: int = None,
    label_fontsize: int = None,
):
    """
    Wrapper function for configuring labels for both axes.
    """
    if title is not None:
        add_titles(axes, title, title_fontsize)
    if xlabel is not None:
        add_x_labels(axes, num_row, num_col, xlabel, label_fontsize)
    if ylabel is not None:
        add_y_labels(axes, num_row, ylabel, label_fontsize)


def prettify_subplots(axes: np.ndarray, num_subplots: int, tick: bool = True, tick_fontsize: int = 12):
    """TODO: docstring"""
    for ax in axes.flat:
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        if tick:
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Remove unused axes entirely
    for _ax in axes.flat[num_subplots:]:
        _ax.remove()


def make_quadratic(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray):
    """
    Utility to make a subplots quadratic in order to avoid visual illusions
    in, e.g., recovery plots.
    """

    lower = min(x_data.min(), y_data.min())
    upper = max(x_data.max(), y_data.max())
    eps = (upper - lower) * 0.1
    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )
