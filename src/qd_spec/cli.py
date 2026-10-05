# Copyright 2026 John Kline
# SPDX-License-Identifier: Apache-2.0

"""Interactive blank-first measurement workflow."""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from . import Spectrometer, fit_spectrum, plot_spectrum, save_measurement


def _confirm(message):
    return input(f"{message} [Y/n] ").strip().casefold() not in {"n", "no"}


def _capture(spec, label):
    input(f"Ready for {label}? Press Enter to continue...")
    print("Reading spectrometer...", flush=True)
    return spec.read()


def _show(figure):
    try:
        plt.show()
    finally:
        plt.close(figure)


def run_cli(base_dir=None, *, integration_time=1000, scans_to_average=10):
    """Acquire one accepted blank, then fit and optionally save each sample."""
    base_dir = Path(base_dir or os.environ.get("QD_SPEC_DATA_DIR") or Path.home() / ".qd_spec")
    run_dir = base_dir / datetime.now(UTC).astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
    count = 0
    blank_saved = False
    print("QD Spectroscopy Measurement Tool")
    with Spectrometer(integration_time=integration_time, scans_to_average=scans_to_average) as spec:
        wavelengths = spec.wavelengths
        while True:
            blank_dark = _capture(spec, "blank dark")
            blank_raw = _capture(spec, "blank (Toluene)")
            _show(plot_spectrum(wavelengths, blank_raw, blank_dark, name="Blank"))
            if _confirm("Accept this blank for the session?"):
                break

        while True:
            name = input("Enter sample name/ID: ").strip()
            dark = _capture(spec, "sample dark")
            raw = _capture(spec, "sample (QDs)")
            print("Fitting spectrum...", flush=True)
            fit = fit_spectrum(wavelengths, raw, dark, blank=blank_raw, blank_dark=blank_dark)
            _show(plot_spectrum(wavelengths, raw, dark, blank=blank_raw, blank_dark=blank_dark, fit=fit, name=name))
            if _confirm("Export this sample?"):
                if not blank_saved:
                    save_measurement(run_dir, "blank", wavelengths, blank_raw, blank_dark)
                    blank_saved = True
                path = save_measurement(run_dir, name, wavelengths, raw, dark, fit=fit)
                print(f"Saved {path}")
            count += 1
            if not _confirm("Measure another sample with the same blank?"):
                break
    print(f"Session completed: {count} sample(s), 1 blank.")


def main():
    parser = argparse.ArgumentParser(description="QD Spectroscopy Measurement Tool")
    parser.add_argument("--data-dir", type=Path, help="Output directory (default: QD_SPEC_DATA_DIR or ~/.qd_spec)")
    parser.add_argument("--integration-time", type=int, default=1000, help="Exposure in milliseconds (default: 1000)")
    parser.add_argument("--scans", type=int, default=10, help="Scans to average (default: 10)")
    args = parser.parse_args()
    try:
        run_cli(args.data_dir, integration_time=args.integration_time, scans_to_average=args.scans)
    except KeyboardInterrupt:
        print("\nSession cancelled.")


if __name__ == "__main__":
    main()
