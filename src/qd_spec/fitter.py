import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


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
    ax.plot(x, y, "b.", label="data")
    ax.plot(
        x,
        single_gauss(x, **{k.rstrip("1"): result.params[k].value for k in ["a1", "x01", "dx1", "yOff"]}),
        "g--",
        label="peak 1",
    )
    ax.plot(
        x,
        single_gauss(x, **{k.rstrip("2"): result.params[k].value for k in ["a2", "x02", "dx2", "yOff"]}),
        "m--",
        label="peak 2",
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
        colors=["g", "m"],
        linestyles="dotted",
        alpha=0.3,
    )
    ax.plot(x, result.best_fit, "r-", label="fit")

    # ax.legend()
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Emission")
    ax.set_xlim(x.min(), x.max())
    fig.show()

    return result
