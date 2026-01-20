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

from .core import QDSession


def main():
    """Interactive CLI for QD measurements with blank-first workflow."""

    def prompt_user(message):
        input(message)

    def confirm_action(message):
        return input(message).casefold() != "n"

    try:
        print("QD Spectroscopy Measurement Tool")
        print("=" * 35)

        # Use QDSession as a context manager to keep spectrometer open
        with QDSession() as session:
            # Step 1: Acquire and validate blank
            print("\nStep 1: Acquiring blank measurement...")
            session.acquire_blank(prompt_func=prompt_user, confirm_func=confirm_action, show_plot=True)
            print("Blank accepted and ready for measurements!")

            # Step 2: Acquire samples
            sample_count = 1
            while True:
                print(f"\nStep 2: Acquiring sample #{sample_count}...")
                try:
                    session.acquire_sample(prompt_func=prompt_user, confirm_func=confirm_action, show_plot=True)
                    print(f"Sample #{sample_count} completed!")
                    sample_count += 1

                    if not confirm_action("\nMeasure another sample with the same blank? [Y/n] "):
                        break

                except KeyboardInterrupt:
                    print("\nSample measurement cancelled.")
                    break

            print(f"\nSession completed! Measured {sample_count - 1} samples with 1 blank.")

    except KeyboardInterrupt:
        print("\nSession cancelled by user.")
    except Exception as e:
        print(f"Error during measurement: {e}")
        raise


if __name__ == "__main__":
    main()
