# Copyright 2026 John Kline
# SPDX-License-Identifier: Apache-2.0

"""Interactive blank-first measurement workflow."""

import argparse
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from . import Spectrometer, fit_spectrum, plot_spectrum, save_measurement

DEFAULT_COM = "COM3"  # Windows default; Linux/Mac will need to specify


def _confirm(message):
    return input(f"{message} [Y/n] ").strip().casefold() not in {"n", "no"}


def _allow_align(laser, label):
    try:
        laser.on()
        input(f"Align the {label}, then press Enter to measure...")
    finally:
        laser.off()


def _capture_pairs(spec, laser, label, pairs, seconds_estimate, samplename=None):
    _allow_align(laser, label)
    darks, lights = [], []
    try:
        with alive_bar(2 * pairs, title=f"{"Sample: " + samplename if samplename else label.title()}") as bar:
            for index in range(1, pairs + 1):
                bar.text = f"Pair {index}/{pairs}: dark"
                laser.off()
                darks.append(spec.read())
                bar()
                bar.text = f"Pair {index}/{pairs}: sample"
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    laser.on()
                lights.append(spec.read())
                bar()
    finally:
        laser.off()
    return np.stack(darks), np.stack(lights)


def _show(figure):
    try:
        plt.show()
    finally:
        plt.close(figure)


def run_cli(base_dir=None, *, integration_time=1000, pairs=10, laser_port=None):
    """Acquire one accepted blank, then fit and optionally save each sample."""
    from coherent_cube import CoherentCube

    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    laser_port = (
        laser_port
        or os.environ.get("COHERENT_CUBE_PORT")
        or input("Laser COM port: ").strip()
    )
    matplotlib.use(os.environ.get("MPLBACKEND", "module://matplotlib-sixel-backend"))
    seconds_estimate = 2 * pairs * integration_time / 1000
    base_dir = Path(
        base_dir or os.environ.get("QD_SPEC_DATA_DIR") or Path.home() / ".qd_spec"
    )
    run_dir = base_dir / datetime.now(UTC).astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
    count = 0
    blank_saved = False
    print("QD Spectroscopy Measurement Tool")
    with CoherentCube(laser_port, timeout=10, write_timeout=10) as laser:
        try:
            laser.cdrh(False)  # CDRH only requires a delay on first startup.
            laser.off()
            with Spectrometer(
                integration_time=integration_time, scans_to_average=1
            ) as spec:
                wavelengths = spec.wavelengths
                while True:
                    blank_dark, blank_raw = _capture_pairs(
                        spec, laser, "blank (Toluene)", pairs, seconds_estimate
                    )
                    _show(
                        plot_spectrum(wavelengths, blank_raw, blank_dark, name="Blank")
                    )
                    if _confirm("Accept this blank for the session?"):
                        break

                while True:
                    name = input("Enter sample name/ID: ").strip()
                    dark, raw = _capture_pairs(
                        spec,
                        laser,
                        "sample (QDs)",
                        pairs,
                        seconds_estimate,
                        samplename=name,
                    )
                    print("Fitting spectrum...", flush=True)
                    fit = fit_spectrum(
                        wavelengths, raw, dark, blank=blank_raw, blank_dark=blank_dark
                    )
                    figure = plot_spectrum(
                        wavelengths,
                        raw,
                        dark,
                        blank=blank_raw,
                        blank_dark=blank_dark,
                        fit=fit,
                        name=name,
                    )
                    _show(figure)
                    if _confirm("Export this sample?"):
                        if not blank_saved:
                            save_measurement(
                                run_dir, "blank", wavelengths, blank_raw, blank_dark
                            )
                            blank_saved = True
                        path = save_measurement(
                            run_dir,
                            name,
                            wavelengths,
                            raw,
                            dark,
                            fit=fit,
                            figure=figure,
                        )
                        print(f"Saved {path}")
                    count += 1
                    if not _confirm("Measure another sample with the same blank?"):
                        break
        finally:
            try:
                laser.off()
            finally:
                laser.cdrh(True)
    print(f"Session completed: {count} sample(s), 1 blank.")


def main():
    parser = argparse.ArgumentParser(description="QD Spectroscopy Measurement Tool")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Output directory (default: QD_SPEC_DATA_DIR or ~/.qd_spec)",
    )
    parser.add_argument(
        "--integration-time",
        type=int,
        default=1000,
        help="Exposure in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--pairs",
        "--scans",
        type=int,
        default=10,
        help="Dark/light pairs per measurement (default: 10)",
    )
    parser.add_argument(
        "--laser-port", help="Laser COM port (default: COHERENT_CUBE_PORT or prompt)"
    )
    args = parser.parse_args()
    try:
        run_cli(
            args.data_dir,
            integration_time=args.integration_time,
            pairs=args.pairs,
            laser_port=args.laser_port
            or os.environ.get("COHERENT_CUBE_PORT")
            or DEFAULT_COM,
        )
    except KeyboardInterrupt:
        print("\nSession cancelled.")


if __name__ == "__main__":
    main()
