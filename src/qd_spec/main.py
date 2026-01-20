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
from .fitter import fit_blank, fit_dg
from .loader import adjust_blank
from .plotter import plot_blank, plot_dg, show_plot


def main():
    with SpectrometerAcquisition() as spec:
        wavs = spec.acquire_wavelengths()
        input("Ready for dark (empty)? Press Enter to continue...")
        blank_dark = spec.acquire_spectrum()
        input("Ready for blank (Toluene)? Press Enter to continue...")
        blank = spec.acquire_spectrum()

        adj_blank = blank - blank_dark
        blank_res = fit_blank(wavs, adj_blank)
        show_plot(plot_blank(blank_res, wavs, blank, blank_dark, adj_blank))

        input("Ready for dark (empty)? Press Enter to continue...")
        sample_dark = spec.acquire_spectrum()
        input("Ready for sample (QDs)? Press Enter to continue...")
        sample = spec.acquire_spectrum()

        adj_sample = sample - sample_dark
        data = adjust_blank(adj_sample, wavs, blank_res)
        res = fit_dg(wavs, data)
        show_plot(plot_dg(res, wavs, sample, sample_dark, data))
        if input("Does this look correct? [Y/n] ").casefold() != "n":
            save_temp(res)
