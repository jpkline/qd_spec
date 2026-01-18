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


def plot(result, x, y):
    fig, ax = plt.subplots()
    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.6,
        color=COLORS["data"],
        linewidths=0,
        label="data",
    )
    ax.plot(
        x,
        single_gauss(x, **{k.rstrip("1"): result.params[k].value for k in ["a1", "x01", "dx1", "yOff"]}),
        linestyle="--",
        color=COLORS["comp1"],
        label="peak 1",
        lw=1.5,
        alpha=0.7,
    )
    ax.plot(
        x,
        single_gauss(x, **{k.rstrip("2"): result.params[k].value for k in ["a2", "x02", "dx2", "yOff"]}),
        linestyle="--",
        color=COLORS["comp2"],
        label="peak 2",
        lw=1.5,
        alpha=0.7,
    )
    ax.vlines(
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

    ax.plot(x, result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
    ax.plot(
        x,
        result.best_fit,
        color=COLORS["fit"],
        label="fit",
        lw=2.5,
    )

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.15)  # TODO: maybe keep grid?

    # ax.legend()
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Emission")
    ax.set_xlim(x.min(), x.max())
    fig.show()
