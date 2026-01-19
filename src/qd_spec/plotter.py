import matplotlib
import matplotlib.pyplot as plt

from .fitter import single_gauss

matplotlib.use("module://matplotlib-sixel-backend")

FG_COLOR = "#F2F2F2"
BG_COLOR = "#0C0C0C"


COLORS = {
    "data": "#5DA9E9",
    "fit": "#E6AF2E",
    "comp1": "#9A6FB0",
    "comp2": "#6FB98F",
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


def plot(result, raw_data, ref, data, x_col, y_col):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.tight_layout(pad=4.0)
    axes[0].set_title("Raw Data")
    axes[1].set_title("Reference")
    axes[2].set_title("Fitted Data")

    axes[0].scatter(
        raw_data[x_col],
        raw_data[y_col],
        s=8,
        # alpha=0.6,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )
    axes[1].scatter(
        ref[x_col],
        ref[y_col],
        s=8,
        # alpha=0.6,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )

    axes[2].scatter(
        data[x_col],
        data[y_col],
        s=8,
        alpha=0.6,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )
    axes[2].plot(
        data[x_col],
        single_gauss(data[x_col], **{k.rstrip("1"): result.params[k].value for k in ["a1", "x01", "dx1", "yOff"]}),
        linestyle="--",
        color=COLORS["comp1"],
        label="peak 1",
        lw=1.5,
        alpha=0.7,
    )
    axes[2].plot(
        data[x_col],
        single_gauss(data[x_col], **{k.rstrip("2"): result.params[k].value for k in ["a2", "x02", "dx2", "yOff"]}),
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

    axes[2].plot(data[x_col], result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
    axes[2].plot(
        data[x_col],
        result.best_fit,
        color=COLORS["fit"],
        label="fit",
        lw=2.5,
    )

    # axes[2].legend()
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.15)  # TODO: maybe keep grid?
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity")
        ax.set_xlim(data[x_col].min(), data[x_col].max())
    fig.show()
