# QD spectroscopy

Acquire Stellarnet spectra, subtract dark and blank readings, fit two Gaussian
peaks, and export plots and CSV data. The API has one stateful class,
`Spectrometer`, and four functions: `correct_spectrum`, `fit_spectrum`,
`plot_spectrum`, and `save_measurement`.

## Install and run

Use Python 3.12 or later for a source installation:

```sh
python -m pip install -e .
qd-spec --data-dir ./measurements
```

Hardware access also requires the vendor's `stellarnet_driverLibs` package and
USB driver. If the package is in a downloaded folder, add its parent directory
to `PYTHONPATH`. Offline fitting and plotting do not need the vendor driver.
The conda export described below bundles the vendor Python library.

The CLI prompts for a dark and solvent blank, lets you accept or retry it, then
captures, fits, and optionally exports each sample. Defaults are 1000 ms and ten
averaged scans; use `--integration-time` and `--scans` to change them. Ctrl+C
ends acquisition and releases the device.

Output defaults to `QD_SPEC_DATA_DIR`, or `~/.qd_spec` if unset. Both
`python -m qd_spec` and `qd_spec` also launch the CLI.

## Python API

```python
from qd_spec import Spectrometer, fit_spectrum, plot_spectrum, save_measurement

with Spectrometer(integration_time=100, scans_to_average=5) as spec:
    wavelengths = spec.wavelengths
    # Arrange the light source/sample before each read.
    input("Ready for dark? ")
    dark = spec.read()
    input("Ready for sample? ")
    raw = spec.read()

result = fit_spectrum(wavelengths, raw, dark)
print(result.fit_report())
figure = plot_spectrum(wavelengths, raw, dark, fit=result, name="sample-1")
save_measurement("measurements/run_1", "sample-1", wavelengths, raw, dark, fit=result, figure=figure)
```

For blank subtraction, pass `blank=blank_raw, blank_dark=blank_dark` to both
`fit_spectrum` and `plot_spectrum`. They apply the same correction internally;
callers do not need to subtract readings themselves. To obtain corrected data
without fitting, use `correct_spectrum(raw, dark, blank=blank_raw,
blank_dark=blank_dark)`. Omitted dark readings default to zero, and omitting
the blank performs dark correction only. `fit_spectrum(wavelengths, corrected)`
still accepts already-corrected data.

Arrays must be one-dimensional and use the same wavelength grid. Plotting
returns a Matplotlib figure; call `plt.show()` to display it.

Fits are ordinary lmfit results. Each `a` is a peak height, each `x0` a center in
nm, and each `dx` a standard deviation in nm; `yOff` is the constant baseline.
The existing bounds remain: amplitudes >= 50, widths >= 15 nm, centers between
100 and 1200 nm. Smoothing only estimates starting centers. Bound warnings can
indicate that the data does not support the assumed peaks.

The CLI defaults to sixel when run. Set `MPLBACKEND` to override it.
Importing the API or CLI does not change Matplotlib's backend.
Acquisition uses the original `alive-progress` spinner from Git history.

## Files and checks

Use one run directory per blank. `save_measurement` writes raw and dark CSVs
with `Wavelength` and `intensity` columns. Pass `figure=...` to save its PNG in
the same directory. With `fit=...`, the CSVs and PNG share the sanitized sample
name and a random ID, and parameters and `_stderr` uncertainties are appended to
`fit_results.csv` in the run directory's parent. Without a fit, it saves a
blank pair. Existing result columns are preserved; use one writer at a time.

```sh
python -m unittest discover -s tests -v
python tests/check_hardware.py --output ./hardware-check
```

The first command uses synthetic data and simulated hardware. The second is an
opt-in check of repeated reads, wavelength calibration, and device reopening.
It saves readings at short and default exposure settings.

The former session, analyzer, profile, acquirer, plotter, and exporter classes
have been removed. Their operations are now explicit function calls, as above.
`Plotter.ipynb` remains a standalone presentation workflow with its own fit bounds.

## Build a conda export

On 64-bit Windows, create and activate a build environment:

```sh
conda create -n qd-build --override-channels -c conda-forge python=3.12 conda-build
conda activate qd-build
python scripts/build_conda.py
```

The script finds `stellarnet_driverLibs` in the project, Python import paths,
or Downloads. Use `--vendor-dir PATH` or `STELLARNET_DRIVER_DIR` to select a
specific folder. It chooses the **highest Python version with a win-64 driver**
(currently 3.12), bundles that binary and `windows_only`, and records their
SHA-256 hashes. The conda package is pinned to that Python minor version.

Required dependencies come directly from `pyproject.toml`. The recipe adds Tk
for desktop plots and libusb for USB access. The default build creates and tests
the application and sixel conda packages. Vendor files are always bundled.
The sixel backend is built from its pinned GitHub revision.

To also resolve, verify, and bundle the complete runtime environment, including
Python and transitive dependencies, opt in with:

```sh
python scripts/build_conda.py --offline-bundle
```

Default builds skip that extra environment creation and ZIP generation. Conda's
package build tests still run.
Each build increments the application build number using packages already in
the output directory, so conda does not reuse an older cached build with the
same version and filename. Keep the output directory between builds.

Outputs:

- `dist/win-64/qd_spec-*.conda`: the application and vendor files, ready to add
  to a conda repository. Dependencies are declared in the package metadata.
- With `--offline-bundle`, `dist/qd_spec-*-win-64-offline.zip`: an indexed local conda channel containing
  the application **and every resolved runtime package**, with exact versions
  and SHA-256 hashes in `packages.json`.

Install from the indexed `dist` channel by package name so conda resolves all
dependencies, including the locally built sixel backend:

```sh
conda install --override-channels -c file:///D:/qd_spec/dist -c conda-forge qd_spec
```

Use your export directory's file URL if it differs. Installing a `.conda` file
directly bypasses dependency resolution. If the archive was installed directly,
ask conda to update the package's dependencies when reinstalling from the channel:

```sh
conda install --update-deps --override-channels -c file:///D:/qd_spec/dist -c conda-forge qd_spec
```

To install without internet, extract the ZIP and run `install.ps1` in an
Anaconda/Miniforge PowerShell prompt. It creates the `qd-spec` environment;
pass `-Name another-name` to choose a different name. Then run
`conda activate qd-spec` and `qd-spec`.

The Windows package runs the bundled `InstallDriver.exe` in a conda post-link
hook when installed. Windows requests administrator approval; accept it to
install the system USB driver. Cancelling or an installer error fails the conda
installation. Set `QD_SPEC_SKIP_DRIVER_INSTALL=1` before installing to skip
driver setup when it is already installed or during automated deployments.
The hook decodes DPInst's status: driver counts are successful results, while
failure flags cause an error. If a reboot is needed, conda displays a message.
The export builder sets this option for its temporary test environments.
The setup program remains available at
`<environment>/Lib/site-packages/stellarnet_driverLibs/windows_only/InstallDriver.exe`.
Vendor files retain their
separate license; see [VENDOR_NOTICE.md](VENDOR_NOTICE.md).

The build never uploads anything. To host the offline channel, extract it into
your repository directory; its `win-64` and `noarch` indexes are already generated.
Use `--output PATH` to choose an export location instead of `dist`.
