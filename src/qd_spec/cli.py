# Copyright 2026 John Kline
# SPDX-License-Identifier: Apache-2.0

"""Interactive blank-first measurement workflow."""

import argparse
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from . import Spectrometer, fit_spectrum, plot_spectrum, save_measurement


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


def _confirm(message):
    return input(f"{message} [Y/n] ").strip().casefold() not in {"n", "no"}


def _capture(spec, label, seconds_estimate):
    input(f"Ready for {label}? Press Enter to continue...")

    def read():
        # Return errors to the main thread so the original spinner can finish.
        try:
            return spec.read()
        except BaseException as error:  # noqa: BLE001 -- includes cancellation in the worker thread
            return error

    result = run_with_spinner(read, seconds_estimate, "Reading Spectrometer")
    if isinstance(result, BaseException):
        raise result
    return result


def _show(figure):
    try:
        plt.show()
    finally:
        plt.close(figure)


def run_cli(base_dir=None, *, integration_time=1000, scans_to_average=10):
    """Acquire one accepted blank, then fit and optionally save each sample."""
    matplotlib.use(os.environ.get("MPLBACKEND", "module://matplotlib-sixel-backend"))
    seconds_estimate = scans_to_average * integration_time / 1000 + 0.5
    base_dir = Path(base_dir or os.environ.get("QD_SPEC_DATA_DIR") or Path.home() / ".qd_spec")
    run_dir = base_dir / datetime.now(UTC).astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
    count = 0
    blank_saved = False
    print("QD Spectroscopy Measurement Tool")
    with Spectrometer(integration_time=integration_time, scans_to_average=scans_to_average) as spec:
        wavelengths = spec.wavelengths
        while True:
            blank_dark = _capture(spec, "blank dark", seconds_estimate)
            blank_raw = _capture(spec, "blank (Toluene)", seconds_estimate)
            _show(plot_spectrum(wavelengths, blank_raw, blank_dark, name="Blank"))
            if _confirm("Accept this blank for the session?"):
                break

        while True:
            name = input("Enter sample name/ID: ").strip()
            dark = _capture(spec, "sample dark", seconds_estimate)
            raw = _capture(spec, "sample (QDs)", seconds_estimate)
            print("Fitting spectrum...", flush=True)
            fit = fit_spectrum(wavelengths, raw, dark, blank=blank_raw, blank_dark=blank_dark)
            figure = plot_spectrum(
                wavelengths, raw, dark, blank=blank_raw, blank_dark=blank_dark, fit=fit, name=name
            )
            _show(figure)
            if _confirm("Export this sample?"):
                if not blank_saved:
                    save_measurement(run_dir, "blank", wavelengths, blank_raw, blank_dark)
                    blank_saved = True
                path = save_measurement(run_dir, name, wavelengths, raw, dark, fit=fit, figure=figure)
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
