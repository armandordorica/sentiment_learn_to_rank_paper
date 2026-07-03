"""Domain-agnostic Plotly chart builders with project hover defaults."""

from __future__ import annotations

import re
from collections.abc import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def melt_wide_metrics(
    df: pd.DataFrame,
    *,
    id_vars: list[str],
    value_vars: list[str],
    series_var: str = "series",
    value_var: str = "value",
    series_name_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Melt wide metric columns into long form (one row per id × series)."""
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=series_var,
        value_name=value_var,
    )
    if series_name_fn is not None:
        long_df[series_var] = long_df[series_var].map(series_name_fn)
    return long_df


def strip_parenthetical_prefix(name: str, prefix: str = "p") -> str:
    """Remove a wrapper like ``p(...)`` from a column name."""
    pattern = rf"^{re.escape(prefix)}\(|\)$"
    return re.sub(pattern, "", str(name))


def apply_category_order(
    df: pd.DataFrame,
    orders: dict[str, list[str]],
) -> pd.DataFrame:
    """Return a copy with categorical column ordering applied."""
    out = df.copy()
    for column, categories in orders.items():
        if column in out.columns:
            out[column] = pd.Categorical(
                out[column], categories=categories, ordered=True
            )
    return out


def group_median(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    output_col: str = "median",
) -> pd.DataFrame:
    """Median of ``value_col`` within each ``group_cols`` group."""
    return (
        df.groupby(group_cols, as_index=False, observed=True)[value_col]
        .median()
        .rename(columns={value_col: output_col})
    )


def _resolve_hover_labels(
    axis_labels: dict[str, str] | None,
    x: str,
    y: str,
    *,
    x_hover_label: str | None = None,
    y_hover_label: str | None = None,
) -> tuple[str, str, str, str]:
    labels = axis_labels or {}
    x_label = labels.get(x, x)
    y_label = labels.get(y, y)
    return x_label, y_label, x_hover_label or x_label, y_hover_label or y_label


def _grouped_series_hovertemplate(
    x_hover: str,
    series_hover_label: str,
    y_hover: str,
    *,
    y_format: str = ".3f",
) -> str:
    return (
        f"{x_hover}: %{{x}}<br>{series_hover_label}: %{{fullData.name}}"
        f"<br>{y_hover}: %{{y:{y_format}}}<extra></extra>"
    )


def _layout_with_y_range(
    y_range: tuple[float, float] | list[float] | None,
    **extra: object,
) -> dict[str, object]:
    layout: dict[str, object] = dict(extra)
    if y_range is not None:
        layout["yaxis_range"] = list(y_range)
    return layout


def _simple_bar_hovertemplate(
    x_hover: str,
    y_hover: str,
    *,
    horizontal: bool = False,
) -> str:
    if horizontal:
        return f"{y_hover}: %{{y}}<br>{x_hover}: %{{x}}<extra></extra>"
    return f"{x_hover}: %{{x}}<br>{y_hover}: %{{y}}<extra></extra>"


def grouped_box_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    category_orders: dict[str, list[str]] | None = None,
    axis_labels: dict[str, str] | None = None,
    y_range: tuple[float, float] | list[float] | None = None,
    points: str = "outliers",
    x_hover_label: str | None = None,
    series_hover_label: str = "Series",
    y_hover_label: str | None = None,
) -> go.Figure:
    """Grouped box-and-whisker chart with closest-hover defaults."""
    _, _, x_hover, y_hover = _resolve_hover_labels(
        axis_labels, x, y, x_hover_label=x_hover_label, y_hover_label=y_hover_label
    )

    fig = px.box(
        df,
        x=x,
        y=y,
        color=series,
        category_orders=category_orders,
        labels=axis_labels,
        title=title,
        points=points,
    )
    fig.update_layout(
        **_layout_with_y_range(y_range, hovermode="closest", boxmode="group")
    )
    fig.update_traces(
        hovertemplate=_grouped_series_hovertemplate(
            x_hover, series_hover_label, y_hover
        )
    )
    return fig


def grouped_median_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    category_orders: dict[str, list[str]] | None = None,
    axis_labels: dict[str, str] | None = None,
    y_range: tuple[float, float] | list[float] | None = None,
    show_value_labels: bool = True,
    value_label_format: str = "%{y:.2f}",
    x_hover_label: str | None = None,
    series_hover_label: str = "Series",
    y_hover_label: str | None = None,
) -> go.Figure:
    """Grouped bar chart for pre-aggregated medians (or other summaries)."""
    _, _, x_hover, y_hover = _resolve_hover_labels(
        axis_labels, x, y, x_hover_label=x_hover_label, y_hover_label=y_hover_label
    )

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=series,
        barmode="group",
        category_orders=category_orders,
        labels=axis_labels,
        title=title,
        text=y if show_value_labels else None,
    )
    fig.update_layout(**_layout_with_y_range(y_range, hovermode="closest"))
    if show_value_labels:
        fig.update_traces(texttemplate=value_label_format, textposition="outside")
    fig.update_traces(
        hovertemplate=_grouped_series_hovertemplate(
            x_hover, series_hover_label, y_hover
        )
    )
    return fig


def split_series_distribution_figures(
    long_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    category_orders: dict[str, list[str]] | None = None,
    axis_labels: dict[str, str] | None = None,
    box_title: str,
    median_title: str,
    median_col: str = "median",
    y_range: tuple[float, float] = (0.0, 1.0),
    x_hover_label: str | None = None,
    series_hover_label: str = "Series",
    y_hover_label: str | None = None,
    median_y_hover_label: str | None = None,
) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    """Build box + median-bar figures from long-form split × series × value data."""
    median_df = group_median(
        long_df,
        group_cols=[x, series],
        value_col=y,
        output_col=median_col,
    )
    fig_box = grouped_box_figure(
        long_df,
        x=x,
        y=y,
        series=series,
        title=box_title,
        category_orders=category_orders,
        axis_labels=axis_labels,
        y_range=y_range,
        x_hover_label=x_hover_label,
        series_hover_label=series_hover_label,
        y_hover_label=y_hover_label,
    )
    fig_median = grouped_median_bar_figure(
        median_df,
        x=x,
        y=median_col,
        series=series,
        title=median_title,
        category_orders=category_orders,
        axis_labels=axis_labels,
        y_range=y_range,
        x_hover_label=x_hover_label,
        series_hover_label=series_hover_label,
        y_hover_label=median_y_hover_label or y_hover_label or median_col,
    )
    return fig_box, fig_median, median_df


def horizontal_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    axis_labels: dict[str, str] | None = None,
    height: int = 240,
    x_hover_label: str | None = None,
    y_hover_label: str | None = None,
) -> go.Figure:
    """Simple horizontal bar chart."""
    _, _, x_hover, y_hover = _resolve_hover_labels(
        axis_labels, x, y, x_hover_label=x_hover_label, y_hover_label=y_hover_label
    )
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        labels=axis_labels,
        title=title,
    )
    fig.update_traces(
        hovertemplate=_simple_bar_hovertemplate(x_hover, y_hover, horizontal=True)
    )
    fig.update_layout(hovermode="closest", showlegend=False, height=height)
    return fig


def vertical_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    axis_labels: dict[str, str] | None = None,
    height: int = 240,
    x_hover_label: str | None = None,
    y_hover_label: str | None = None,
) -> go.Figure:
    """Simple vertical bar chart."""
    _, _, x_hover, y_hover = _resolve_hover_labels(
        axis_labels, x, y, x_hover_label=x_hover_label, y_hover_label=y_hover_label
    )
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        labels=axis_labels,
        title=title,
    )
    fig.update_traces(
        hovertemplate=_simple_bar_hovertemplate(x_hover, y_hover, horizontal=False)
    )
    fig.update_layout(hovermode="closest", showlegend=False, height=height)
    return fig
