"""Opt-in smoke test: python tests/check_hardware.py --output ./hardware-check.

Reads the connected device without assuming different light/dark conditions.
Use PYTHONPATH to expose src and the parent of stellarnet_driverLibs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qd_spec import Spectrometer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for integration_time, scans in [(50, 1), (1000, 10)]:
        with Spectrometer(integration_time=integration_time, scans_to_average=scans) as spectrometer:
            wavelengths = spectrometer.wavelengths
            assert wavelengths.ndim == 1 and wavelengths.size > 1
            assert np.isfinite(wavelengths).all()
            assert (np.diff(wavelengths) > 0).all(), "Wavelength grid is not increasing"
            for reading in range(3):
                started = time.perf_counter()
                intensity = spectrometer.read()
                elapsed = time.perf_counter() - started
                assert intensity.shape == wavelengths.shape, "Wavelength/intensity lengths differ"
                assert np.isfinite(intensity).all(), "Invalid intensity values"
                pd.DataFrame({"Wavelength": wavelengths, "intensity": intensity}).to_csv(
                    args.output / f"{integration_time}ms_{scans}scans_{reading}.csv", index=False
                )
                entry = {
                    "integration_ms": integration_time,
                    "averaged_scans": scans,
                    "reading": reading,
                    "points": intensity.size,
                    "wavelength_min_nm": float(wavelengths.min()),
                    "wavelength_max_nm": float(wavelengths.max()),
                    "intensity_min": float(intensity.min()),
                    "intensity_max": float(intensity.max()),
                    "seconds": round(elapsed, 3),
                }
                summary.append(entry)
                print(json.dumps(entry), flush=True)
        print("Device reset successfully", flush=True)
    # Reopening also verifies that reset releases the device for subsequent use.
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
