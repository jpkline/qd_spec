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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from stellarnet_driverLibs import stellarnet_driver3 as sn

matplotlib.use("module://matplotlib-sixel-backend")


@dataclass(frozen=True)
class SpectrometerSettings:
    """Configuration applied to the Stellarnet spectrometer."""

    integration_time: int = 1000
    scans_to_average: int = 1  # Increase after stabilizing firmware.
    smoothing: int = 0
    xtiming: int = 3
    channel: int = 0
    use_external_trigger: bool = False


@dataclass(frozen=True)
class PlotTheme:
    """Declarative matplotlib theme so styling is centralized and testable."""

    foreground: str = "#F2F2F2"
    background: str = "#0C0C0C"
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "data": "#5DA9E9",
            "fit": "#E6AF2E",
            "comp1": "#C97C5D",
            "comp2": "#C6A0CF",
            "baseline": "#888888",
        }
    )

    def apply(self) -> None:
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": self.background,
                "axes.facecolor": self.background,
                "savefig.facecolor": self.background,
                "text.color": self.foreground,
                "axes.labelcolor": self.foreground,
                "axes.titlecolor": self.foreground,
                "xtick.color": self.foreground,
                "ytick.color": self.foreground,
                "axes.edgecolor": self.foreground,
                "lines.dash_capstyle": "round",
                "lines.solid_capstyle": "round",
            }
        )


THEME = PlotTheme()
THEME.apply()


class Spectrometer:
    """Hardware interface for spectrometer acquisition."""

    def __init__(self, settings: SpectrometerSettings | None = None):
        self.settings = settings or SpectrometerSettings()
        self._spec = sn.array_get_spec_only(self.settings.channel)
        sn.ext_trig(self._spec, self.settings.use_external_trigger)
        sn.setParam(
            self._spec,
            self.settings.integration_time,
            self.settings.scans_to_average,
            self.settings.smoothing,
            self.settings.xtiming,
            clear=True,
        )

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


class GaussianProfile(ABC):
    """Base class for parameterized Gaussian models with lmfit integration."""

    def __init__(self, name: str, evaluator: Callable[..., np.ndarray]):
        self.name = name
        self._model = lmfit.Model(evaluator)
        self._evaluator = evaluator

    def fit(self, x: np.ndarray, y: np.ndarray) -> lmfit.model.ModelResult:
        params = self._initial_params(x, y)
        result = self._model.fit(y, params, x=x)
        print(result.fit_report())
        self._warn_on_bounds(result)
        return result

    def evaluate(self, x: np.ndarray, **params) -> np.ndarray:
        return self._evaluator(x, **params)

    def evaluate_from_result(self, result: lmfit.model.ModelResult, x: np.ndarray, **overrides) -> np.ndarray:
        params = {name: param.value for name, param in result.params.items()}
        params.update(overrides)
        return self.evaluate(x, **params)

    def _warn_on_bounds(self, result: lmfit.model.ModelResult) -> None:
        for name, info in result.params.items():
            if np.isclose(info.value, info.min) or np.isclose(info.value, info.max):
                print(f"WARNING: Parameter `{name}` hit bound: {info.value:.2f}")

    @abstractmethod
    def _initial_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        """Return initial parameters tuned to the incoming spectrum."""


def _single_gauss(x, a, x0, dx, yOff):
    return a * np.exp(-((x - x0) ** 2) / (2 * dx**2)) + yOff


class SingleGaussianProfile(GaussianProfile):
    """Single-peak Gaussian profile for blank fitting."""

    def __init__(self):
        super().__init__("single_gaussian", _single_gauss)

    def _initial_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        params = self._model.make_params(a=10, x0=700, dx=20, yOff=5)
        params["a"].set(min=0.01)
        params["dx"].set(min=10)
        params["x0"].set(min=100, max=1200, value=x[gaussian_filter1d(y, sigma=5).argmax()])
        return params


def _double_gauss(x, a1, x01, dx1, a2, x02, dx2, yOff):
    term1 = a1 * np.exp(-((x - x01) ** 2) / (2 * dx1**2))
    term2 = a2 * np.exp(-((x - x02) ** 2) / (2 * dx2**2))
    return term1 + term2 + yOff


class DoubleGaussianProfile(GaussianProfile):
    """Double-peak Gaussian profile for QD sample fitting."""

    def __init__(self):
        super().__init__("double_gaussian", _double_gauss)

    def _initial_params(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
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


@dataclass
class BlankMeasurement:
    """Blank measurement that will be used for all samples in a session."""

    wavelengths: np.ndarray
    blank_raw: np.ndarray
    blank_dark: np.ndarray
    blank_fit: lmfit.model.ModelResult
    model: GaussianProfile

    @property
    def blank_corrected(self) -> np.ndarray:
        return self.blank_raw - self.blank_dark


@dataclass
class SampleMeasurement:
    """Individual sample measurement using a shared blank."""

    sample_raw: np.ndarray
    sample_dark: np.ndarray
    sample_fit: lmfit.model.ModelResult
    blank: BlankMeasurement
    model: GaussianProfile

    @property
    def sample_corrected(self) -> np.ndarray:
        return self.sample_raw - self.sample_dark

    @property
    def sample_blank_adjusted(self) -> np.ndarray:
        """Sample with blank subtracted (zero offset)."""
        blank_signal = self.blank.model.evaluate_from_result(self.blank.blank_fit, self.blank.wavelengths, yOff=0)
        return self.sample_corrected - blank_signal

    @property
    def wavelengths(self) -> np.ndarray:
        return self.blank.wavelengths


class MeasurementExporter(ABC):
    """Strategy interface for persisting measurement metadata."""

    @abstractmethod
    def export(self, sample: SampleMeasurement) -> None:
        raise NotImplementedError


class QDPlotter:
    """Plotting interface for QD measurements."""

    def __init__(self, theme: PlotTheme = THEME):
        self.theme = theme

    def _create_base_layout(
        self,
        wavelengths: np.ndarray,
        raw_data: np.ndarray,
        dark_data: np.ndarray,
        processed_data: np.ndarray,
        titles: list[str],
    ):
        palette = self.theme.palette
        fig = plt.figure(figsize=(8, 7))
        gs = GridSpec(2, 2, height_ratios=[2, 3])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

        for i, (ax, title, data, size) in enumerate(
            zip(axes, titles, [raw_data, dark_data, processed_data], [4, 4, 8])
        ):
            ax.set_title(title)
            ax.scatter(
                wavelengths,
                data,
                s=size,
                alpha=0.8 if i < 2 else 0.6,
                color=palette["data"],
                linewidths=0,
            )
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.15)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Intensity")
            ax.set_xlim(wavelengths.min(), wavelengths.max())

        return fig, axes

    def plot_blank(self, blank: BlankMeasurement):
        palette = self.theme.palette
        fig, axes = self._create_base_layout(
            blank.wavelengths,
            blank.blank_raw,
            blank.blank_dark,
            blank.blank_corrected,
            ["Blank", "Dark", "Fitted Blank"],
        )
        result = blank.blank_fit
        fitted_peak = blank.model.evaluate_from_result(result, blank.wavelengths)
        axes[2].plot(
            blank.wavelengths,
            fitted_peak,
            linestyle="--",
            color=palette["comp1"],
            label="peak",
            lw=1.5,
            alpha=0.7,
        )
        center = result.params["x0"].value
        center_idx = (np.abs(blank.wavelengths - center)).argmin()
        axes[2].vlines(
            center,
            result.params["yOff"].value,
            fitted_peak[center_idx],
            colors=palette["comp1"],
            linestyles=":",
            linewidths=1.2,
            alpha=0.8,
        )
        axes[2].plot(blank.wavelengths, result.best_fit, color=palette["fit"], lw=5, alpha=0.15)
        axes[2].plot(blank.wavelengths, result.best_fit, color=palette["fit"], label="fit", lw=2.5)
        return fig, axes

    def plot_sample(self, sample: SampleMeasurement):
        palette = self.theme.palette
        fig, axes = self._create_base_layout(
            sample.wavelengths,
            sample.sample_raw,
            sample.sample_dark,
            sample.sample_blank_adjusted,
            ["Sample", "Dark", "Fitted Data"],
        )
        result = sample.sample_fit
        for i, color in enumerate([palette["comp1"], palette["comp2"]], start=1):
            component = _single_gauss(
                sample.wavelengths,
                a=result.params[f"a{i}"].value,
                x0=result.params[f"x0{i}"].value,
                dx=result.params[f"dx{i}"].value,
                yOff=result.params["yOff"].value,
            )
            axes[2].plot(
                sample.wavelengths,
                component,
                linestyle="--",
                color=color,
                label=f"peak {i}",
                lw=1.5,
                alpha=0.7,
            )
            center_idx = (np.abs(sample.wavelengths - result.params[f"x0{i}"].value)).argmin()
            axes[2].vlines(
                result.params[f"x0{i}"].value,
                result.params["yOff"].value,
                component[center_idx],
                colors=color,
                linestyles=":",
                linewidths=1.2,
                alpha=0.8,
            )

        axes[2].plot(sample.wavelengths, result.best_fit, color=palette["fit"], lw=5, alpha=0.15)
        axes[2].plot(sample.wavelengths, result.best_fit, color=palette["fit"], label="fit", lw=2.5)
        return fig, axes

    def show(self, fig_axes):
        fig, _ = fig_axes
        fig.tight_layout(pad=1.0)
        fig.show()


class QDAnalyzer:
    """Core analysis engine for QD measurements."""

    def __init__(
        self,
        blank_profile: GaussianProfile | None = None,
        sample_profile: GaussianProfile | None = None,
    ):
        self.blank_profile = blank_profile or SingleGaussianProfile()
        self.sample_profile = sample_profile or DoubleGaussianProfile()

    def analyze_blank(self, wavelengths: np.ndarray, blank_dark: np.ndarray, blank_raw: np.ndarray) -> BlankMeasurement:
        blank_corrected = blank_raw - blank_dark
        blank_fit = self.blank_profile.fit(wavelengths, blank_corrected)
        return BlankMeasurement(
            wavelengths=wavelengths,
            blank_raw=blank_raw,
            blank_dark=blank_dark,
            blank_fit=blank_fit,
            model=self.blank_profile,
        )

    def analyze_sample(
        self,
        blank: BlankMeasurement,
        sample_dark: np.ndarray,
        sample_raw: np.ndarray,
    ) -> SampleMeasurement:
        sample_corrected = sample_raw - sample_dark
        blank_zeroed = blank.model.evaluate_from_result(blank.blank_fit, blank.wavelengths, yOff=0)
        sample_adjusted = sample_corrected - blank_zeroed
        sample_fit = self.sample_profile.fit(blank.wavelengths, sample_adjusted)
        return SampleMeasurement(
            sample_raw=sample_raw,
            sample_dark=sample_dark,
            sample_fit=sample_fit,
            blank=blank,
            model=self.sample_profile,
        )


@dataclass
class ResultExporter(MeasurementExporter):
    """Export measurement results to CSV."""

    path: str = "temp_results.csv"

    def export(self, sample: SampleMeasurement) -> None:
        result = sample.sample_fit
        new_data = pd.DataFrame({k: v.value for k, v in result.params.items()}, index=[0])

        try:
            df = pd.read_csv(self.path)
            df = pd.concat([df, new_data], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = new_data

        df.to_csv(self.path, index=False)


class QDSession:
    """Session manager for QD measurements with blank-first workflow."""

    def __init__(
        self,
        settings: SpectrometerSettings | None = None,
        export_path: str = "temp_results.csv",
        exporter: MeasurementExporter | None = None,
        plotter: QDPlotter | None = None,
        analyzer: QDAnalyzer | None = None,
    ):
        self.settings = settings or SpectrometerSettings()
        self.spectrometer: Spectrometer | None = None
        self.analyzer = analyzer or QDAnalyzer()
        self.plotter = plotter or QDPlotter()
        self.exporter = exporter or ResultExporter(export_path)
        self.blank: BlankMeasurement | None = None

    def __enter__(self):
        self.spectrometer = Spectrometer(self.settings)
        self.spectrometer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spectrometer:
            self.spectrometer.__exit__(exc_type, exc_val, exc_tb)
            self.spectrometer = None

    def acquire_blank(
        self,
        prompt_func: Callable[[str], object] | None = None,
        confirm_func: Callable[[str], bool] | None = None,
        show_plot: bool = True,
    ) -> BlankMeasurement:
        spec = self._require_active_spectrometer()
        prompt = prompt_func or (lambda *_: None)
        confirm = confirm_func or (lambda *_: True)

        wavelengths = spec.acquire_wavelengths()
        prompt("Ready for blank dark? Press Enter to continue...")
        blank_dark = spec.acquire_spectrum()

        prompt("Ready for blank (Toluene)? Press Enter to continue...")
        blank_raw = spec.acquire_spectrum()

        blank = self.analyzer.analyze_blank(wavelengths, blank_dark, blank_raw)
        if show_plot:
            self.plotter.show(self.plotter.plot_blank(blank))

        if confirm("Accept this blank for the session? [Y/n] "):
            self.blank = blank
            return blank
        raise ValueError("Blank rejected - please run acquire_blank() again")

    def acquire_sample(
        self,
        prompt_func: Callable[[str], object] | None = None,
        confirm_func: Callable[[str], bool] | None = None,
        show_plot: bool = True,
    ) -> SampleMeasurement:
        spec = self._require_active_spectrometer()
        if self.blank is None:
            raise ValueError("No blank available - call acquire_blank() first")

        prompt = prompt_func or (lambda *_: None)
        confirm = confirm_func or (lambda *_: True)

        prompt("Ready for sample dark? Press Enter to continue...")
        sample_dark = spec.acquire_spectrum()

        prompt("Ready for sample (QDs)? Press Enter to continue...")
        sample_raw = spec.acquire_spectrum()

        sample = self.analyzer.analyze_sample(self.blank, sample_dark, sample_raw)
        if show_plot:
            self.plotter.show(self.plotter.plot_sample(sample))

        if confirm("Export this sample? [Y/n] "):
            self.exporter.export(sample)
        return sample

    @property
    def has_blank(self) -> bool:
        """Check if session has a validated blank."""
        return self.blank is not None

    @property
    def is_active(self) -> bool:
        """Check if session has an active spectrometer connection."""
        return self.spectrometer is not None

    def _require_active_spectrometer(self) -> Spectrometer:
        if self.spectrometer is None:
            raise ValueError("Session not active - use 'with QDSession() as session:'")
        return self.spectrometer
