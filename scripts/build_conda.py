"""Build a Windows conda package and an offline bundle of all runtime dependencies.

The newest CPython ABI in the local Stellarnet driver folder selects the target
Python version. Run with Python from a conda environment containing conda-build.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]


def driver_version(path):
    """Read the numeric CPython version from a Windows 64-bit extension name."""
    match = re.fullmatch(r"stellarnet_driver3\.cp(\d)(\d+)-win_amd64\.pyd", path.name)
    return tuple(map(int, match.groups())) if match else (0, 0)


def find_vendor(explicit=None):
    """Find the highest-version driver, honoring an explicit folder first."""
    configured = explicit or os.environ.get("STELLARNET_DRIVER_DIR")
    if configured:
        folders = [Path(configured).expanduser()]
    else:
        folders = [ROOT / "vendor" / "stellarnet_driverLibs", ROOT / "stellarnet_driverLibs"]
        folders += [Path(path) / "stellarnet_driverLibs" for path in sys.path if path]
        folders.append(Path.home() / "Downloads" / "stellarnet_driverLibs")
    drivers = [
        path
        for folder in folders
        if (folder / "windows_only" / "InstallDriver.exe").is_file()
        for path in folder.glob("stellarnet_driver3.cp*-win_amd64.pyd")
        if driver_version(path) != (0, 0)
    ]
    if not drivers:
        raise FileNotFoundError(
            "Cannot find a Windows 64-bit StellarNet driver and windows_only/InstallDriver.exe. "
            "Pass --vendor-dir PATH or set STELLARNET_DRIVER_DIR to the stellarnet_driverLibs folder."
        )
    return max(drivers, key=driver_version).resolve()


def stage_source(destination, driver):
    """Copy release sources and the selected vendor files, with SHA-256 hashes."""
    shutil.copytree(
        ROOT / "src" / "qd_spec", destination / "src" / "qd_spec", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "conda", destination / "conda")
    (destination / "tests").mkdir()
    shutil.copy2(ROOT / "tests" / "test_workflow.py", destination / "tests")
    for name in ("pyproject.toml", "README.md", "LICENSE.md", "VENDOR_NOTICE.md"):
        shutil.copy2(ROOT / name, destination / name)
    python = ".".join(map(str, driver_version(driver)))
    (destination / "vendor-info.json").write_text(json.dumps({"python": python}), encoding="utf-8")
    bundled = destination / "vendor" / "stellarnet_driverLibs"
    bundled.mkdir(parents=True)
    shutil.copy2(driver, bundled / driver.name)
    shutil.copytree(driver.parent / "windows_only", bundled / "windows_only")
    manifest = {
        path.relative_to(bundled).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundled.rglob("*"))
        if path.is_file()
    }
    (bundled / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def bundle_environment(prefix, channel, archive):
    """Copy exact resolved conda archives into a portable, indexed local channel."""
    from conda_index.api import update_index

    manifest = []
    for metadata in sorted((prefix / "conda-meta").glob("*.json")):
        record = json.loads(metadata.read_text(encoding="utf-8"))
        cached = Path(record["link"]["source"]).parent / record["fn"]
        destination = channel / record["subdir"] / record["fn"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if record.get("sha256") and digest != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {cached}")
        manifest.append(
            {
                "name": record["name"],
                "version": record["version"],
                "build": record["build"],
                "file": destination.relative_to(channel).as_posix(),
                "sha256": digest,
            }
        )
    update_index(str(channel), threads=1, verbose=False, progress=False)
    (channel / "packages.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (channel / "install.ps1").write_text(
        'param([string]$Name = "qd-spec")\n'
        "$channel = ([System.Uri]::new($PSScriptRoot + [IO.Path]::DirectorySeparatorChar)).AbsoluteUri\n"
        "conda create --name $Name --offline --solver classic --override-channels --channel $channel qd_spec -y\n"
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
        'Write-Host "Run: conda activate $Name"\n',
        encoding="utf-8",
    )
    (channel / "README.txt").write_text(
        "Unzip this folder, then run install.ps1 from an Anaconda/Miniforge PowerShell prompt.\n"
        "The bundle includes Python and all required runtime dependencies; no internet is needed.\n"
        "packages.json records exact versions and SHA-256 hashes.\n"
        "The required sixel backend is included; set MPLBACKEND=module://matplotlib-sixel-backend to use it.\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for path in sorted(channel.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                output.write(path, path.relative_to(channel))
    print(f"Offline bundle: {archive} ({len(manifest)} packages)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", type=Path, help="Folder containing the vendor's Python drivers")
    parser.add_argument("--output", type=Path, default=ROOT / "dist", help="Output directory (default: dist)")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.maxsize <= 2**32:
        parser.error("This export targets Windows 64-bit; build it on 64-bit Windows.")
    if importlib.util.find_spec("conda_build") is None:
        parser.error("Run this script with Python from a conda environment containing conda-build.")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    driver = find_vendor(args.vendor_dir)
    python = ".".join(map(str, driver_version(driver)))
    if python not in SpecifierSet(project["requires-python"]):
        parser.error(
            f"Newest vendor driver targets Python {python}, but qd_spec requires {project['requires-python']}."
        )
    print(f"Targeting Python {python}; bundling {driver}", flush=True)
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("CONDA_PKGS_DIRS", str(build_dir / "conda-pkgs"))
    env.update(
        PYTHONIOENCODING="utf-8",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(build_dir / "matplotlib"),
        QD_SPEC_SKIP_DRIVER_INSTALL="1",
    )
    conda = [sys.executable, "-m", "conda"]
    with tempfile.TemporaryDirectory(prefix="export-", dir=build_dir) as temporary:
        source = Path(temporary)
        stage_source(source, driver)
        build_command = conda + [
            "build",
            "--python",
            python,
            "--package-format",
            "2",
            "--no-anaconda-upload",
            "--override-channels",
            "-c",
            output.as_uri(),
            "-c",
            "conda-forge",
            "--croot",
            str(build_dir / "conda"),
            "--output-folder",
            str(output),
        ]
        for recipe in (source / "conda" / "sixel", source / "conda"):
            subprocess.run(build_command + [str(recipe)], check=True, env=env)
        prefix = source / "runtime"
        subprocess.run(
            conda
            + [
                "create",
                "--prefix",
                str(prefix),
                "--override-channels",
                "-c",
                output.as_uri(),
                "-c",
                "conda-forge",
                f"qd_spec={project['version']}",
                "--quiet",
                "-y",
            ],
            check=True,
            env=env,
        )
        # Test the resolved environment, not just the package build environment.
        runtime_env = {
            **env,
            "PATH": os.pathsep.join(str(prefix / part) for part in ("", "Library/bin", "Scripts"))
            + os.pathsep
            + env.get("PATH", ""),
        }
        subprocess.run(
            [
                str(prefix / "python.exe"),
                "-c",
                "import importlib, qd_spec, lmfit; importlib.import_module('matplotlib-sixel-backend')",
            ],
            check=True,
            env=runtime_env,
        )
        subprocess.run([str(prefix / "Scripts" / "qd-spec.exe"), "--help"], check=True, env=runtime_env)
        archive = output / f"qd_spec-{project['version']}-py{python.replace('.', '')}-win-64-offline.zip"
        bundle_environment(prefix, source / "offline", archive)


if __name__ == "__main__":
    main()
