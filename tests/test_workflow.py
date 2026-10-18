"""Regression checks using synthetic spectra and a simulated vendor driver."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alive_progress import alive_bar

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qd_spec import Spectrometer, correct_spectrum, fit_spectrum, plot_spectrum, save_measurement
from qd_spec.cli import run_cli


class WorkflowTests(unittest.TestCase):
    def test_cli_backend_changes_only_when_run_and_honors_override(self):
        from qd_spec import cli

        with patch("matplotlib.use") as use:
            importlib.reload(cli)
            use.assert_not_called()
            for override, expected in ((None, "module://matplotlib-sixel-backend"), ("Agg", "Agg")):
                with (
                    patch.dict(os.environ),
                    patch("builtins.input", side_effect=KeyboardInterrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    os.environ.pop("MPLBACKEND", None)
                    if override:
                        os.environ["MPLBACKEND"] = override
                    cli.run_cli(laser_port="COM1")
                use.assert_called_with(expected)

    def test_cli_propagates_read_failure_and_releases_device(self):
        self.driver.getSpectrum_Y.side_effect = RuntimeError("read failed")
        with patch("builtins.input", return_value=""), self.assertRaisesRegex(RuntimeError, "read failed"):
            run_cli(laser_port="COM1")
        self.driver.reset.assert_called_once()
        self.laser.off.assert_called()

    def test_failed_laser_on_leaves_laser_off(self):
        self.laser.on.side_effect = RuntimeError("laser failed")
        with patch("builtins.input", return_value=""), self.assertRaisesRegex(RuntimeError, "laser failed"):
            run_cli(laser_port="COM1")
        self.assertGreaterEqual(self.laser.off.call_count, 2)
        self.driver.reset.assert_called_once()

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
        self.laser = Mock()
        self.laser.__enter__ = Mock(return_value=self.laser)
        self.laser.__exit__ = Mock(return_value=None)
        self.laser_factory = Mock(return_value=self.laser)
        self.vendor_patch = patch.dict(
            "sys.modules",
            {
                "stellarnet_driverLibs": vendor,
                "coherent_cube": SimpleNamespace(CoherentCube=self.laser_factory),
            },
        )
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

    def test_fit_and_main_plot_use_mean_of_corrected_pairs(self):
        dark = np.stack([self.dark + index for index in range(10)])
        blank_dark = np.stack([self.dark + 2 * index for index in range(10)])
        blank = blank_dark + 20 + np.arange(10)[:, None]
        raw = dark + blank - blank_dark + self.signal + np.arange(10)[:, None]
        corrected = correct_spectrum(raw, dark, blank=blank, blank_dark=blank_dark)
        np.testing.assert_allclose(corrected, self.signal + np.arange(10)[:, None])
        fit = fit_spectrum(self.x, raw, dark, blank=blank, blank_dark=blank_dark)
        np.testing.assert_allclose(fit.best_fit, corrected.mean(axis=0), atol=1e-3)
        figure = plot_spectrum(self.x, raw, dark, blank=blank, blank_dark=blank_dark, fit=fit)
        self.assertEqual(len(figure.axes[2].collections), 1)
        np.testing.assert_allclose(figure.axes[0].collections[0].get_offsets()[:, 1], blank.mean(axis=0))
        np.testing.assert_allclose(figure.axes[1].collections[0].get_offsets()[:, 1], raw.mean(axis=0))
        np.testing.assert_allclose(figure.axes[2].collections[0].get_offsets()[:, 1], corrected.mean(axis=0))

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
                figure = plot_spectrum(self.x, self.signal, self.dark, fit=result, name=name)
                save_measurement(run_dir, name, self.x, self.signal + self.blank, self.dark, fit=result, figure=figure)
                plt.close(figure)
            files = list(run_dir.glob("*.csv"))
            self.assertEqual(len(files), 6)
            self.assertEqual(len(list(run_dir.glob("*.png"))), 2)
            rows = pd.read_csv(results_path)
            self.assertEqual(rows.sample_name.tolist(), ["older", "first", "second"])
            self.assertEqual(rows.legacy[0], 42)
            self.assertIn("a1_stderr", rows.columns)
            self.assertEqual(rows.sample_uid.nunique(), 2)
            for path in files:
                self.assertEqual(pd.read_csv(path).columns.tolist(), ["Wavelength", "intensity"])

    def test_exports_every_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            readings = np.stack([self.signal + index for index in range(10)])
            darks = np.stack([self.dark + index for index in range(10)])
            path = save_measurement(directory, "paired", self.x, readings, darks, fit=fit_spectrum(self.x, self.signal))
            self.assertTrue(path.name.endswith("_sample_01.csv"))
            self.assertEqual(len(list(Path(directory).glob("*_sample_*.csv"))), 10)
            self.assertEqual(len(list(Path(directory).glob("*_dark_*.csv"))), 10)
            np.testing.assert_allclose(pd.read_csv(path).intensity, readings[0])
            np.testing.assert_allclose(
                pd.read_csv(path.with_name(path.name.replace("_sample_01", "_dark_01"))).intensity, darks[0]
            )

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
        readings = []
        for offset in (100, 0, 0, 0):
            for index in range(2):
                dark = self.dark + index
                light = dark + 20 + offset
                if len(readings) >= 8:
                    light = light + self.signal
                readings.extend((dark, light))
        self.laser.lit = True
        self.laser.off.side_effect = lambda: setattr(self.laser, "lit", False)
        self.laser.on.side_effect = lambda: setattr(self.laser, "lit", True)
        reading_order = iter(enumerate(readings))

        def read_at_laser_state(_device):
            index, reading = next(reading_order)
            self.assertEqual(self.laser.lit, bool(index % 2))
            return reading

        self.driver.getSpectrum_Y.side_effect = read_at_laser_state
        answers = ["", "n", "", "y", "first", "", "n", "y", "second", "", "y", "n"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("builtins.input", side_effect=answers),
            patch("matplotlib.pyplot.show"),
            patch("qd_spec.cli.alive_bar", wraps=alive_bar) as progress,
        ):
            run_cli(directory, pairs=2, laser_port="COM1")
            self.assertEqual(progress.call_count, 4)
            for call in progress.call_args_list:
                self.assertEqual(call.args, (4,))
            rows = pd.read_csv(Path(directory) / "fit_results.csv")
            self.assertEqual(rows.sample_name.tolist(), ["second"])
            self.assertAlmostEqual(rows.x01[0], 650, places=3)
            run_dir = next(Path(directory).glob("run_*"))
            self.assertEqual(len(list(run_dir.glob("*.csv"))), 8)
            plot_path = next(run_dir.glob("second_*.png"))
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertFalse(list(run_dir.glob("first_*.png")))
            np.testing.assert_array_equal(pd.read_csv(run_dir / "blank_blank_01.csv").intensity, readings[5])
        self.driver.reset.assert_called_once()
        self.driver.setParam.assert_called_once_with(
            self.driver.array_get_spec_only.return_value, 1000, 1, 0, 3, clear=True
        )
        self.laser_factory.assert_called_once_with("COM1", timeout=10, write_timeout=10)
        self.assertEqual(self.driver.getSpectrum_Y.call_count, 16)
        self.assertEqual(self.laser.off.call_count, 18)
        self.assertEqual(self.laser.on.call_count, 12)

    def test_cli_interruption_releases_device(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            run_cli(laser_port="COM1")
        self.driver.reset.assert_called_once()
        self.assertEqual(self.laser.off.call_count, 3)


if __name__ == "__main__":
    unittest.main()
