# Copyright 2026 John Kline
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QD spectroscopy library: acquisition, fitting, plotting, and data export."""

from __future__ import annotations

from dataclasses import dataclass

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from stellarnet_driverLibs import stellarnet_driver3 as sn

matplotlib.use("module://matplotlib-sixel-backend")

# Hardware constants
INTTIME = 1000
SCANSAVG = 1  # FIXME: 10
SMOOTH = 0
XTIMING = 3
CHANNEL = 0

# Plot styling
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
        "text.color": FG_COLOR,
        "axes.labelcolor": FG_COLOR,
        "axes.titlecolor": FG_COLOR,
        "xtick.color": FG_COLOR,
        "ytick.color": FG_COLOR,
        "axes.edgecolor": FG_COLOR,
        "lines.dash_capstyle": "round",
        "lines.solid_capstyle": "round",
    }
)


# Mathematical models
def double_gauss(x, a1, x01, dx1, a2, x02, dx2, yOff):
    return a1 * np.exp(-((x - x01) ** 2) / (2 * dx1**2)) + a2 * np.exp(-((x - x02) ** 2) / (2 * dx2**2)) + yOff


def single_gauss(x, a, x0, dx, yOff):
    return a * np.exp(-((x - x0) ** 2) / (2 * dx**2)) + yOff


@dataclass
class QDMeasurement:
    """Complete QD measurement with all data and fit results."""

    wavelengths: np.ndarray
    blank_raw: np.ndarray
    blank_dark: np.ndarray
    blank_fit: lmfit.model.ModelResult
    sample_raw: np.ndarray
    sample_dark: np.ndarray
    sample_fit: lmfit.model.ModelResult

    @property
    def blank_corrected(self) -> np.ndarray:
        return self.blank_raw - self.blank_dark

    @property
    def sample_corrected(self) -> np.ndarray:
        return self.sample_raw - self.sample_dark

    @property
    def sample_blank_adjusted(self) -> np.ndarray:
        """Sample with blank subtracted (zero offset)."""
        params = {k: (v if k != "yOff" else 0) for k, v in self.blank_fit.params.items()}
        return self.sample_corrected - single_gauss(self.wavelengths, **params)


class Spectrometer:
    """Hardware interface for spectrometer acquisition."""

    def __init__(self, channel: int = CHANNEL):
        self._spec = sn.array_get_spec_only(channel)
        sn.ext_trig(self._spec, False)
        sn.setParam(self._spec, INTTIME, SCANSAVG, SMOOTH, XTIMING, clear=True)

    def acquire_spectrum(self) -> np.ndarray:
        return sn.getSpectrum_Y(self._spec)

    def acquire_wavelengths(self) -> np.ndarray:
        return sn.getSpectrum_X(self._spec)

    def close(self) -> None:
        sn.reset(self._spec)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _BaseFit:
    """Base class for Gaussian fitting."""

    def __init__(self, model_func):
        self._model = lmfit.Model(model_func)

    def _setup_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> lmfit.model.ModelResult:
        params = self._setup_params(x, y)
        result = self._model.fit(y, params, x=x)
        print(result.fit_report())

        # Check bounds
        for name, info in result.params.items():
            if np.isclose(info.value, info.min) or np.isclose(info.value, info.max):
                print(f"WARNING: Parameter `{name}` hit bound: {info.value:.2f}")

        return result


class BlankFit(_BaseFit):
    """Single Gaussian fit for blank spectra."""

    def __init__(self):
        super().__init__(single_gauss)

    def _setup_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        params = self._model.make_params(a=10, x0=700, dx=20, yOff=5)
        params["a"].set(min=0.01)
        params["dx"].set(min=10)
        params["x0"].set(min=100, max=1200, value=x[gaussian_filter1d(y, sigma=5).argmax()])
        return params


class SampleFit(_BaseFit):
    """Double Gaussian fit for QD sample spectra."""

    def __init__(self):
        super().__init__(double_gauss)

    def _setup_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        params = self._model.make_params(a1=10, x01=500, dx1=20, a2=10, x02=600, dx2=20, yOff=10)
        params["a1"].set(min=50)
        params["a2"].set(min=50)
        params["dx1"].set(min=15)
        params["dx2"].set(min=15)
        params["x01"].set(min=100, max=1200)
        params["x02"].set(min=100, max=1200)

        peak_est = x[gaussian_filter1d(y, sigma=5).argmax()]
        params["x01"].set(value=peak_est + 50)
        params["x02"].set(value=peak_est - 50)
        return params


class QDAnalyzer:
    """Core analysis engine for QD measurements."""

    def __init__(self):
        self.blank_fitter = BlankFit()
        self.sample_fitter = SampleFit()

    def analyze(
        self,
        wavelengths: np.ndarray,
        blank_dark: np.ndarray,
        blank_raw: np.ndarray,
        sample_dark: np.ndarray,
        sample_raw: np.ndarray,
    ) -> QDMeasurement:
        # Fit blank
        blank_corrected = blank_raw - blank_dark
        blank_fit = self.blank_fitter.fit(wavelengths, blank_corrected)

        # Process sample
        sample_corrected = sample_raw - sample_dark
        params = {k: (v if k != "yOff" else 0) for k, v in blank_fit.params.items()}
        sample_adjusted = sample_corrected - single_gauss(wavelengths, **params)
        sample_fit = self.sample_fitter.fit(wavelengths, sample_adjusted)

        return QDMeasurement(
            wavelengths=wavelengths,
            blank_raw=blank_raw,
            blank_dark=blank_dark,
            blank_fit=blank_fit,
            sample_raw=sample_raw,
            sample_dark=sample_dark,
            sample_fit=sample_fit,
        )


class QDPlotter:
    """Plotting interface for QD measurements."""

    def _create_base_layout(self, measurement: QDMeasurement, is_blank: bool = True):
        fig = plt.figure(figsize=(8, 7))
        gs = GridSpec(2, 2, height_ratios=[2, 3])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

        if is_blank:
            raw_data, dark_data, processed_data = (
                measurement.blank_raw,
                measurement.blank_dark,
                measurement.blank_corrected,
            )
            titles = ["Blank", "Dark", "Fitted Blank"]
        else:
            raw_data, dark_data, processed_data = (
                measurement.sample_raw,
                measurement.sample_dark,
                measurement.sample_blank_adjusted,
            )
            titles = ["Sample", "Dark", "Fitted Data"]

        for i, (ax, title, data, size) in enumerate(
            zip(axes, titles, [raw_data, dark_data, processed_data], [4, 4, 8])
        ):
            ax.set_title(title)
            ax.scatter(
                measurement.wavelengths, data, s=size, alpha=0.8 if i < 2 else 0.6, color=COLORS["data"], linewidths=0
            )
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.15)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Intensity")
            ax.set_xlim(measurement.wavelengths.min(), measurement.wavelengths.max())

        return fig, axes

    def plot_blank(self, measurement: QDMeasurement):
        fig, axes = self._create_base_layout(measurement, is_blank=True)
        result = measurement.blank_fit

        # Add fit components
        axes[2].plot(
            measurement.wavelengths,
            single_gauss(measurement.wavelengths, **{k: result.params[k].value for k in ["a", "x0", "dx", "yOff"]}),
            linestyle="--",
            color=COLORS["comp1"],
            label="peak",
            lw=1.5,
            alpha=0.7,
        )
        axes[2].vlines(
            result.params["x0"].value,
            result.params["yOff"].value,
            single_gauss(result.params["x0"].value, **{k: result.params[k].value for k in ["a", "x0", "dx", "yOff"]}),
            colors=COLORS["comp1"],
            linestyles=":",
            linewidths=1.2,
            alpha=0.8,
        )

        # Add overall fit
        axes[2].plot(measurement.wavelengths, result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
        axes[2].plot(measurement.wavelengths, result.best_fit, color=COLORS["fit"], label="fit", lw=2.5)

        return fig, axes

    def plot_sample(self, measurement: QDMeasurement):
        fig, axes = self._create_base_layout(measurement, is_blank=False)
        result = measurement.sample_fit

        # Add individual peaks
        for i, (suffix, color) in enumerate([("1", COLORS["comp1"]), ("2", COLORS["comp2"])], 1):
            peak_params = {k.rstrip(suffix): result.params[k].value for k in [f"a{i}", f"x0{i}", f"dx{i}", "yOff"]}
            axes[2].plot(
                measurement.wavelengths,
                single_gauss(measurement.wavelengths, **peak_params),
                linestyle="--",
                color=color,
                label=f"peak {i}",
                lw=1.5,
                alpha=0.7,
            )

        # Add peak markers
        x_centers = [result.params[f"x0{i}"].value for i in [1, 2]]
        y_centers = [
            single_gauss(x, **{k.rstrip(str(i)): result.params[k].value for k in [f"a{i}", f"x0{i}", f"dx{i}", "yOff"]})
            for i, x in enumerate(x_centers, 1)
        ]
        axes[2].vlines(
            x_centers,
            [result.params["yOff"].value] * 2,
            y_centers,
            colors=[COLORS["comp1"], COLORS["comp2"]],
            linestyles=":",
            linewidths=1.2,
            alpha=0.8,
        )

        # Add overall fit
        axes[2].plot(measurement.wavelengths, result.best_fit, color=COLORS["fit"], lw=5, alpha=0.15)
        axes[2].plot(measurement.wavelengths, result.best_fit, color=COLORS["fit"], label="fit", lw=2.5)

        return fig, axes

    def show(self, fig_axes):
        fig, _ = fig_axes
        fig.tight_layout(pad=1.0)
        fig.show()


@dataclass
class ResultExporter:
    """Export measurement results to CSV."""

    path: str = "temp_results.csv"

    def export(self, measurement: QDMeasurement) -> None:
        result = measurement.sample_fit
        new_data = pd.DataFrame({k: v.value for k, v in result.params.items()}, index=[0])

        try:
            df = pd.read_csv(self.path)
            df = pd.concat([df, new_data], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = new_data

        df.to_csv(self.path, index=False)


class QDSession:
    """High-level session manager for QD measurements."""

    def __init__(self, channel: int = CHANNEL, export_path: str = "temp_results.csv"):
        self.spectrometer = Spectrometer(channel)
        self.analyzer = QDAnalyzer()
        self.plotter = QDPlotter()
        self.exporter = ResultExporter(export_path)

    def acquire_spectra(self, prompt_func=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Acquire all required spectra with optional prompting."""

        def silent_prompt(msg):
            pass

        if prompt_func is None:
            prompt_func = silent_prompt

        with self.spectrometer as spec:
            wavelengths = spec.acquire_wavelengths()

            prompt_func("Ready for dark (empty)? Press Enter to continue...")
            blank_dark = spec.acquire_spectrum()

            prompt_func("Ready for blank (Toluene)? Press Enter to continue...")
            blank_raw = spec.acquire_spectrum()

            prompt_func("Ready for dark (empty)? Press Enter to continue...")
            sample_dark = spec.acquire_spectrum()

            prompt_func("Ready for sample (QDs)? Press Enter to continue...")
            sample_raw = spec.acquire_spectrum()

        return wavelengths, blank_dark, blank_raw, sample_dark, sample_raw

    def run_measurement(self, prompt_func=None, confirm_func=None, show_plots: bool = True) -> QDMeasurement:
        """Run a complete measurement workflow."""
        # Acquire data
        wavelengths, blank_dark, blank_raw, sample_dark, sample_raw = self.acquire_spectra(prompt_func)

        # Analyze
        measurement = self.analyzer.analyze(wavelengths, blank_dark, blank_raw, sample_dark, sample_raw)

        # Show plots
        if show_plots:
            self.plotter.show(self.plotter.plot_blank(measurement))
            self.plotter.show(self.plotter.plot_sample(measurement))

        # Export if confirmed
        if confirm_func is None or confirm_func("Does this look correct? [Y/n] "):
            self.exporter.export(measurement)

        return measurement
