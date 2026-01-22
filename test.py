# %%
import threading
import time

# %%
from alive_progress import alive_bar


def run_with_spinner(func, seconds_estimate, title):
    done = threading.Event()
    result = {}

    def worker():
        result["value"] = func()
        done.set()

    t = threading.Thread(target=worker)
    t.start()

    with alive_bar(total=0, manual=True, title=title) as bar:
        t0 = time.perf_counter()
        while 1:
            if done.is_set():
                bar(percent=1.0)
                break
            time.sleep(0.012)
            bar(percent=(time.perf_counter() - t0) / seconds_estimate)

    t.join()
    return result["value"]


# %%
exposure_s = 10

# %%
run_with_spinner(lambda: time.sleep(exposure_s), exposure_s, "Reading Spectrometer")

# %%

import numpy as np
from lmfit.models import GaussianModel


def callback_model(params, iter, resid, *args, **kws):
    print(f"Iteration {iter}: {params['height'].value}")


# Setup model and data
x = np.linspace(0, 20, 1000000)
y = 1.5 * np.exp(-((x - 10) ** 2) / (2 * 1**2)) + np.random.normal(0, 0.1, x.size)

gmodel = GaussianModel()
params = gmodel.make_params(height=2, fwhm=5, sigma=2)

with alive_bar(title="Fitting", unit=" iterations") as bar:
    result = gmodel.fit(y, params, x=x, iter_cb=lambda *a, **k: bar())
