# Copyright 2026 John Kline
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for QD spectroscopy measurements."""

from __future__ import annotations

from collections.abc import Callable

from .core import QDSession


class MeasurementCLI:
    """Interactive CLI orchestrated with explicit object state."""

    def __init__(self, session_factory: Callable[[], QDSession] = QDSession):
        self._session_factory = session_factory

    def run(self) -> None:
        self._print_banner()
        with self._session_factory() as session:
            samples_measured = self._acquire_samples(session)
        self._print_summary(samples_measured)

    def _prompt(self, message: str) -> str:
        return input(message)

    def _pause(self, message: str) -> None:
        self._prompt(message)

    def _confirm(self, message: str) -> bool:
        return self._prompt(message).strip().casefold() not in {"n", "no"}

    def _print_banner(self) -> None:
        print("QD Spectroscopy Measurement Tool")
        print("=" * 35)

    def _print_summary(self, samples_measured: int) -> None:
        print(f"\nSession completed! Measured {samples_measured} samples.")

    def _acquire_samples(self, session: QDSession) -> int:
        sample_count = 0
        while True:
            sample_number = sample_count + 1
            print(f"\nAcquiring sample #{sample_number}...")
            acquirer = session.create_sample_acquirer(show_plot=True)
            try:
                self._pause("Ready for sample dark? Press Enter to continue...")
                acquirer.capture_dark()
                self._pause("Ready for sample (QDs)? Press Enter to continue...")
                measurement = acquirer.capture_sample()
            except KeyboardInterrupt:
                print("\nSample measurement cancelled.")
                break

            if self._confirm("Export this sample? [Y/n] "):
                session.export_sample(measurement)

            print(f"Sample #{sample_number} completed!")
            sample_count += 1

            if not self._confirm("\nMeasure another sample? [Y/n] "):
                break

        return sample_count


def run_cli() -> None:
    MeasurementCLI().run()


def main() -> None:
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nSession cancelled by user.")
    except Exception as exc:  # pragma: no cover - surfaced for operator visibility
        print(f"Error during measurement: {exc}")
        raise


if __name__ == "__main__":
    main()
