# Copyright 2026 John Kline

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .fitter import single_gauss

matplotlib.use("module://matplotlib-sixel-backend")

FG_COLOR = "#F2F2F2"
BG_COLOR = "#0C0C0C"

COLORS = {
    "data": "#5DA9E9",
    "fit": "#E6AF2E",
    "comp1": "#C97C5D",
    "comp2": "#C6A0CF",
    "baseline": "#888888",
}


plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        #
        "text.color": FG_COLOR,
        "axes.labelcolor": FG_COLOR,
        "axes.titlecolor": FG_COLOR,
        "xtick.color": FG_COLOR,
        "ytick.color": FG_COLOR,
        "axes.edgecolor": FG_COLOR,
    }
)
plt.rcParams["lines.dash_capstyle"] = "round"
plt.rcParams["lines.solid_capstyle"] = "round"


def plot_dg(result, wavs, raw_data, ref, data):
    fig, axes = plot_nofit(wavs, raw_data, ref, data, t1="Sample", t3="Fitted Data")
    axes[2].plot(
        wavs,
        single_gauss(wavs, **{k.rstrip("1"): result.params[k].value for k in ["a1", "x01", "dx1", "yOff"]}),
        linestyle="--",
        color=COLORS["comp1"],
        label="peak 1",
        lw=1.5,
        alpha=0.7,
    )
    axes[2].plot(
        wavs,
        single_gauss(wavs, **{k.rstrip("2"): result.params[k].value for k in ["a2", "x02", "dx2", "yOff"]}),
        linestyle="--",
        color=COLORS["comp2"],
        label="peak 2",
        lw=1.5,
        alpha=0.7,
    )
    axes[2].vlines(
        [result.params["x01"].value, result.params["x02"].value],
        ymin=[result.params["yOff"].value] * 2,
        ymax=[
            single_gauss(
                result.params["x01"].value,
                **{k.rstrip("1"): result.params[k].value for k in ["a1", "x01", "dx1", "yOff"]},
            ),
            single_gauss(
                result.params["x02"].value,
                **{k.rstrip("2"): result.params[k].value for k in ["a2", "x02", "dx2", "yOff"]},
            ),
        ],
        colors=[COLORS["comp1"], COLORS["comp2"]],
        linestyles=":",
        linewidths=1.2,
        alpha=0.8,
    )

    axes[2].plot(wavs, result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
    axes[2].plot(
        wavs,
        result.best_fit,
        color=COLORS["fit"],
        label="fit",
        lw=2.5,
    )
    return fig, axes


def plot_blank(result, wavs, raw_data, ref, data):
    fig, axes = plot_nofit(wavs, raw_data, ref, data, t1="Blank", t3="Fitted Blank")
    axes[2].plot(
        wavs,
        single_gauss(wavs, **{k: result.params[k].value for k in ["a", "x0", "dx", "yOff"]}),
        linestyle="--",
        color=COLORS["comp1"],
        label="peak 1",
        lw=1.5,
        alpha=0.7,
    )
    axes[2].vlines(
        [result.params["x0"].value],
        ymin=[result.params["yOff"].value],
        ymax=[
            single_gauss(
                result.params["x0"].value,
                **{k: result.params[k].value for k in ["a", "x0", "dx", "yOff"]},
            ),
        ],
        colors=[COLORS["comp1"], COLORS["comp2"]],
        linestyles=":",
        linewidths=1.2,
        alpha=0.8,
    )

    axes[2].plot(wavs, result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
    axes[2].plot(
        wavs,
        result.best_fit,
        color=COLORS["fit"],
        label="fit",
        lw=2.5,
    )
    return fig, axes


def plot_nofit(wavs, raw, dark, data, t1="Blank", t3="Adjusted Blank"):
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 2, height_ratios=[2, 3])  # bottom is taller

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    axes[0].set_title(t1)
    axes[1].set_title("Dark")
    axes[2].set_title(t3)

    axes[0].scatter(
        wavs,
        raw,
        s=4,
        alpha=0.8,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )
    axes[1].scatter(
        wavs,
        dark,
        s=4,
        alpha=0.8,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )

    axes[2].scatter(
        wavs,
        data,
        s=8,
        alpha=0.6,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.15)  # TODO: maybe keep grid?
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity")
        ax.set_xlim(wavs.min(), wavs.max())

    return fig, axes


def show_plot(fax):
    fig, _axes = fax
    fig.tight_layout(pad=1.0)
    fig.show()
