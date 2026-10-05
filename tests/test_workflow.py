"""Regression checks using synthetic spectra and a simulated vendor driver."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qd_spec import Spectrometer, correct_spectrum, fit_spectrum, plot_spectrum, save_measurement
from qd_spec.cli import run_cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(400, 800, 401)
        self.dark = np.full_like(self.x, 5)
        self.blank = self.dark + 20
        self.signal = 200 * np.exp(-0.5 * ((self.x - 650) / 25) ** 2)
        self.signal += 120 * np.exp(-0.5 * ((self.x - 550) / 20) ** 2) + 3
        self.driver = Mock()
        self.driver.getSpectrum_X.return_value = self.x.tolist()
        self.driver.getSpectrum_Y.return_value = self.signal.tolist()
        vendor = SimpleNamespace(stellarnet_driver3=self.driver)
        self.vendor_patch = patch.dict("sys.modules", {"stellarnet_driverLibs": vendor})
        self.vendor_patch.start()
        self.addCleanup(self.vendor_patch.stop)
        self.addCleanup(plt.close, "all")

    def test_fit_and_plot(self):
        result = fit_spectrum(self.x, self.signal + self.blank, self.dark, blank=self.blank, blank_dark=self.dark)
        np.testing.assert_allclose(result.best_fit, self.signal, atol=1e-4)
        self.assertAlmostEqual(result.best_values["x01"], 650, places=3)
        self.assertAlmostEqual(result.best_values["x02"], 550, places=3)
        before = plt.rcParams["axes.facecolor"]
        fig = plot_spectrum(
            self.x,
            self.signal + self.blank,
            self.dark,
            blank=self.blank,
            blank_dark=self.dark,
            fit=result,
            name="Synthetic",
        )
        self.assertEqual(len(fig.axes), 3)
        np.testing.assert_allclose(fig.axes[2].collections[0].get_offsets()[:, 1], self.signal)
        fig.canvas.draw()
        self.assertEqual(plt.rcParams["axes.facecolor"], before)

    def test_correction_handles_unsigned_counts_without_changing_inputs(self):
        raw = np.array([10, 20], dtype=np.uint16)
        dark = np.array([15, 5], dtype=np.uint16)
        blank = np.array([8, 4], dtype=np.uint16)
        blank_dark = np.array([3, 9], dtype=np.uint16)
        readings = (raw, dark, blank, blank_dark)
        originals = [reading.copy() for reading in readings]
        np.testing.assert_array_equal(correct_spectrum(raw, dark), [-5, 15])
        np.testing.assert_array_equal(correct_spectrum(raw, dark, blank=blank, blank_dark=blank_dark), [-10, 20])
        np.testing.assert_array_equal(correct_spectrum(raw), raw)
        for reading, original in zip(readings, originals):
            np.testing.assert_array_equal(reading, original)

    def test_exports_preserve_existing_columns_and_uncertainties(self):
        result = fit_spectrum(self.x, self.signal)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run_test"
            results_path = Path(directory) / "fit_results.csv"
            pd.DataFrame({"sample_name": ["older"], "legacy": [42]}).to_csv(results_path, index=False)
            save_measurement(run_dir, "blank", self.x, self.blank, self.dark)
            for name in ("first", "second"):
                save_measurement(run_dir, name, self.x, self.signal + self.blank, self.dark, fit=result)
            files = list(run_dir.glob("*.csv"))
            self.assertEqual(len(files), 6)
            rows = pd.read_csv(results_path)
            self.assertEqual(rows.sample_name.tolist(), ["older", "first", "second"])
            self.assertEqual(rows.legacy[0], 42)
            self.assertIn("a1_stderr", rows.columns)
            self.assertEqual(rows.sample_uid.nunique(), 2)
            for path in files:
                self.assertEqual(pd.read_csv(path).columns.tolist(), ["Wavelength", "intensity"])

    def test_hardware_settings_and_close(self):
        with Spectrometer(integration_time=50, scans_to_average=2) as spec:
            self.driver.setParam.assert_called_once_with(
                self.driver.array_get_spec_only.return_value, 50, 2, 0, 3, clear=True
            )
            np.testing.assert_array_equal(spec.wavelengths, self.x)
            np.testing.assert_array_equal(spec.read(), self.signal)
        spec.close()
        self.driver.reset.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            spec.read()

    def test_failed_setup_releases_device(self):
        self.driver.setParam.side_effect = RuntimeError("configuration failed")
        with self.assertRaisesRegex(RuntimeError, "configuration failed"):
            Spectrometer()
        self.driver.reset.assert_called_once()

    def test_failed_read_releases_device(self):
        self.driver.getSpectrum_Y.side_effect = RuntimeError("read failed")
        with self.assertRaisesRegex(RuntimeError, "read failed"), Spectrometer() as spec:
            spec.read()
        self.driver.reset.assert_called_once()

    def test_cli_retries_blank_and_only_exports_selected_samples(self):
        self.driver.getSpectrum_Y.side_effect = [
            self.dark,
            self.blank + 100,
            self.dark,
            self.blank,
            self.dark,
            self.signal + self.blank,
            self.dark,
            self.signal + self.blank,
        ]
        answers = ["", "", "n", "", "", "y", "first", "", "", "n", "y", "second", "", "", "y", "n"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("builtins.input", side_effect=answers),
            patch("matplotlib.pyplot.show"),
        ):
            run_cli(directory)
            rows = pd.read_csv(Path(directory) / "fit_results.csv")
            self.assertEqual(rows.sample_name.tolist(), ["second"])
            self.assertAlmostEqual(rows.x01[0], 650, places=3)
            run_dir = next(Path(directory).glob("run_*"))
            self.assertEqual(len(list(run_dir.glob("*.csv"))), 4)
            np.testing.assert_array_equal(pd.read_csv(run_dir / "blank_blank.csv").intensity, self.blank)
        self.driver.reset.assert_called_once()
        self.assertEqual(self.driver.getSpectrum_Y.call_count, 8)

    def test_cli_interruption_releases_device(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            run_cli()
        self.driver.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
