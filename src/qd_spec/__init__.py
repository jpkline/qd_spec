# Copyright 2026 John Kline
# SPDX-License-Identifier: Apache-2.0

"""Acquire spectra, fit two Gaussian peaks, plot, and export measurements."""

from .core import Spectrometer, correct_spectrum, fit_spectrum, plot_spectrum, save_measurement

__all__ = ["Spectrometer", "correct_spectrum", "fit_spectrum", "plot_spectrum", "save_measurement"]
