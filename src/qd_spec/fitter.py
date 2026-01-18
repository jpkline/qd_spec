import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

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


def double_gauss(x, a1, x01, dx1, a2, x02, dx2, yOff):
    return a1 * np.exp(-((x - x01) ** 2) / (2 * dx1**2)) + a2 * np.exp(-((x - x02) ** 2) / (2 * dx2**2)) + yOff


def single_gauss(x, a, x0, dx, yOff):
    return a * np.exp(-((x - x0) ** 2) / (2 * dx**2)) + yOff


def fit_dg(x, y):
    model = lmfit.Model(double_gauss)
    params = model.make_params(a1=10, x01=500, dx1=20, a2=10, x02=600, dx2=20, yOff=10)

    params["a1"].set(min=50)
    params["a2"].set(min=50)
    params["dx1"].set(min=15)
    params["dx2"].set(min=15)
    params["x01"].set(min=100, max=1200)
    params["x02"].set(min=100, max=1200)

    peak_est = x[gaussian_filter1d(y, sigma=5).argmax()]
    params["x01"].set(value=peak_est + 50)
    params["x02"].set(value=peak_est - 50)

    result = model.fit(y, params, x=x)

    print(result.fit_report())

    for param, info in result.params.items():
        if np.isclose(info.value, info.min) or np.isclose(info.value, info.max):
            print(
                f"WARNING: Parameter `{param}` hit a bound: value={info.value:.2f}, min={info.min:.2f}, max={info.max:.2f}"
            )

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

    return result
