# Copyright 2026 John Kline
# SPDX-License-Identifier: Apache-2.0

"""Spectrometer access, two-Gaussian fitting, plotting, and CSV export."""

import uuid
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathvalidate import sanitize_filename
from scipy.ndimage import gaussian_filter1d


class Spectrometer:
    """Open a Stellarnet device; use a with-block to release it reliably.

    Integration time is in milliseconds. Each read returns the average of
    ``scans_to_average`` scans on the device's fixed ``wavelengths`` grid (nm).
    """

    def __init__(
        self,
        *,
        integration_time=1000,
        scans_to_average=10,
        smoothing=0,
        xtiming=3,
        channel=0,
        use_external_trigger=False,
    ):
        # Keep the vendor dependency out of offline analysis.
        from stellarnet_driverLibs import stellarnet_driver3

        self._driver = stellarnet_driver3
        self._device = self._driver.array_get_spec_only(channel)
        try:
            self._driver.ext_trig(self._device, use_external_trigger)
            self._driver.setParam(self._device, integration_time, scans_to_average, smoothing, xtiming, clear=True)
            self.wavelengths = np.asarray(self._driver.getSpectrum_X(self._device))
        except BaseException:
            self.close()
            raise

    def read(self) -> np.ndarray:
        """Read one spectrum using the configured averaging and exposure."""
        if self._device is None:
            raise RuntimeError("Spectrometer is closed")
        return np.asarray(self._driver.getSpectrum_Y(self._device))

    def close(self) -> None:
        """Release the device. Calling close more than once is harmless."""
        if self._device is not None:
            device, self._device = self._device, None
            self._driver.reset(device)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def _double_gaussian(x, a1, x01, dx1, a2, x02, dx2, yOff):
    # Parameter names are kept consistent with existing fit_results.csv files.
    return _gaussian(x, a1, x01, dx1) + _gaussian(x, a2, x02, dx2) + yOff


def correct_spectrum(raw, dark=0, *, blank=None, blank_dark=0) -> np.ndarray:
    """Return (raw - dark) - (blank - blank_dark), without modifying inputs.

    Omit the blank for dark correction only. Omitted dark readings are zero.
    Convert counts to floats before subtracting to avoid unsigned underflow.
    All supplied spectra must use the same wavelength grid.
    """
    corrected = np.asarray(raw, dtype=float) - np.asarray(dark, dtype=float)
    if blank is not None:
        corrected -= np.asarray(blank, dtype=float) - np.asarray(blank_dark, dtype=float)
    return corrected


def fit_spectrum(wavelengths, raw, dark=0, *, blank=None, blank_dark=0) -> lmfit.model.ModelResult:
    """Correct the readings, then fit two Gaussians and a constant baseline.

    Supply raw blank and dark readings, or omit them for already-corrected data.
    Amplitudes are peak heights; widths are standard deviations in nm.
    Smoothing is only used to guess peak centers; the fit uses original data.
    """
    intensity = correct_spectrum(raw, dark, blank=blank, blank_dark=blank_dark)
    model = lmfit.Model(_double_gaussian)
    params = model.make_params(a1=50, x01=500, dx1=20, a2=50, x02=600, dx2=20, yOff=10)
    peak = wavelengths[gaussian_filter1d(intensity, sigma=5).argmax()]
    for i, center in enumerate((peak + 50, peak - 50), start=1):
        params[f"a{i}"].set(min=50)
        params[f"dx{i}"].set(min=15)
        params[f"x0{i}"].set(value=center, min=100, max=1200)
    result = model.fit(intensity, params, x=wavelengths)
    for name, param in result.params.items():
        if np.isclose(param.value, param.min) or np.isclose(param.value, param.max):
            print(f"WARNING: Parameter `{name}` hit bound: {param.value:.2f}")
    return result


def plot_spectrum(wavelengths, raw, dark, *, blank=None, blank_dark=0, fit=None, name="Spectrum"):
    """Return a figure with raw, dark, and corrected spectra.

    Pass the same raw blank and dark readings as for fit_spectrum. ``fit`` is
    its result. Display with plt.show() or save with figure.savefig(path).
    """
    corrected = correct_spectrum(raw, dark, blank=blank, blank_dark=blank_dark)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplot_mosaic(
            [["raw", "dark"], ["corrected", "corrected"]],
            height_ratios=[2, 3],
            figsize=(8, 7),
            layout="constrained",
        )
        for ax, title, data in zip(axes.values(), (name, "Dark", "Corrected spectrum"), (raw, dark, corrected)):
            ax.scatter(wavelengths, data, s=5, alpha=0.6, color="#5DA9E9", linewidths=0)
            ax.set(
                title=title, xlabel="Wavelength [nm]", ylabel="Intensity", xlim=(wavelengths.min(), wavelengths.max())
            )
            ax.grid(alpha=0.15)
        if fit is not None:
            ax = axes["corrected"]
            params = fit.best_values
            for i, color in enumerate(("#C97C5D", "#C6A0CF"), start=1):
                component = _gaussian(wavelengths, params[f"a{i}"], params[f"x0{i}"], params[f"dx{i}"])
                ax.plot(wavelengths, component + params["yOff"], "--", color=color, label=f"Peak {i}")
            ax.plot(wavelengths, fit.best_fit, color="#E6AF2E", label="Fit")
            ax.legend()
    return fig


def save_measurement(run_dir, name, wavelengths, raw, dark, *, fit=None, figure=None) -> Path:
    """Save raw and dark CSVs, an optional plot, and optional fit results.

    Use one run directory per blank. Without a fit, filenames end in _blank
    and _blank_dark. Samples get a random ID linking their files to the row
    in run_dir.parent / 'fit_results.csv'. Returns the raw-spectrum path.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(name) or "unnamed"
    sample_uid = uuid.uuid4().hex[:8]
    stem = f"{safe_name}_{sample_uid}" if fit is not None else f"{safe_name}_blank"
    raw_path = run_dir / f"{stem}_sample.csv" if fit is not None else run_dir / f"{stem}.csv"
    for path, intensity in ((raw_path, raw), (run_dir / f"{stem}_dark.csv", dark)):
        pd.DataFrame({"Wavelength": wavelengths, "intensity": intensity}).to_csv(path, index=False)
    if figure is not None:
        figure.savefig(run_dir / f"{stem}.png")
    if fit is not None:
        row = {"sample_uid": sample_uid}
        for key, param in fit.params.items():
            row[key] = param.value
            row[f"{key}_stderr"] = param.stderr
        rows = pd.DataFrame(row, index=pd.Index([name], name="sample_name"))
        results_path = run_dir.parent / "fit_results.csv"
        try:
            rows = pd.concat([pd.read_csv(results_path, index_col=0), rows])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        rows.to_csv(results_path)
    return raw_path
