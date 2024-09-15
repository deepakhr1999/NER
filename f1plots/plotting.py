import plotly.graph_objects as go
import numpy as np
import webcolors
import pandas as pd


def to_opacity(name, opacity):
    r, g, b = webcolors.name_to_rgb(name)
    return f"rgba({r}, {g}, {b}, {opacity})"


def update_layout(
    fig: go.Figure,
    title: str,
    xlabel: str,
    ylabel: str,
    upkwargs: dict = {},
    uxkwargs: dict = {},
    uykwargs: dict = {},
    row=None,
    col=None,
):
    fig.update_layout(
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        autosize=False,
        # width=571,
        # height=457,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        title={
            "text": title,
            "x": 0.53,
            "font": {"size": 15},
        },
        **upkwargs,
    )
    fig.update_xaxes(
        title_text=xlabel,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        row=row,
        col=col,
        **uxkwargs,
    )
    fig.update_yaxes(
        title_text=ylabel,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        row=row,
        col=col,
        **uykwargs,
    )