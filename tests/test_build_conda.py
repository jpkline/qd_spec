"""Check export staging without downloading packages or opening hardware."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_conda import find_vendor, stage_source


class ExportTests(unittest.TestCase):
    def test_newest_abi_is_selected_and_only_needed_files_are_staged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vendor = root / "vendor"
            (vendor / "windows_only").mkdir(parents=True)
            (vendor / "windows_only" / "InstallDriver.exe").write_bytes(b"installer")
            for abi in ("39", "310", "311", "312"):
                (vendor / f"stellarnet_driver3.cp{abi}-win_amd64.pyd").write_bytes(abi.encode())
            (vendor / "stellarnet_driver3.cp313-win32.pyd").write_bytes(b"wrong platform")
            selected = find_vendor(vendor)
            self.assertEqual(selected.name, "stellarnet_driver3.cp312-win_amd64.pyd")
            staged = root / "source"
            stage_source(staged, selected)
            self.assertEqual(json.loads((staged / "vendor-info.json").read_text())["python"], "3.12")
            bundled = staged / "vendor" / "stellarnet_driverLibs"
            self.assertEqual([path.name for path in bundled.glob("*.pyd")], [selected.name])
            manifest = json.loads((bundled / "manifest.json").read_text())
            self.assertEqual(manifest[selected.name], hashlib.sha256(b"312").hexdigest())
            self.assertTrue((staged / "tests" / "test_workflow.py").exists())
            self.assertFalse(list(staged.rglob("*.ipynb")))

    def test_explicit_missing_vendor_does_not_fall_back_to_another_installation(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(FileNotFoundError, "--vendor-dir"):
            find_vendor(Path(temporary))


if __name__ == "__main__":
    unittest.main()
