import lmfit
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

    return result
