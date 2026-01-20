# Copyright 2026 John Kline

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .acquisition import SpectrometerAcquisition
from .exporter import save_temp
from .fitter import fit_dg
from .loader import adjust_ref
from .plotter import plot


def main():
    with SpectrometerAcquisition() as spec:
        wavs = spec.acquire_wavelengths()
        input("Ready for dark (empty)? Press Enter to continue...")
        blank_dark = spec.acquire_spectrum()
        input("Ready for blank (Toluene)? Press Enter to continue...")
        blank = spec.acquire_spectrum()
        input("Ready for dark (empty)? Press Enter to continue...")
        sample_dark = spec.acquire_spectrum()
        input("Ready for sample (QDs)? Press Enter to continue...")
        sample = spec.acquire_spectrum()

        adj_blank = adjust_ref(blank, blank_dark)
        adj_sample = adjust_ref(sample, sample_dark)
        data = adjust_ref(adj_sample, adj_blank)
        res = fit_dg(wavs, data)
        plot(res, wavs, sample, sample_dark, data)
        if input("Does this look correct? [Y/n] ").casefold() != "n":
            save_temp(res)
