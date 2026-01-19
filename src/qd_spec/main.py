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

from .fitter import fit_dg
from .loader import adjust_ref, load_data
from .plotter import plot


def main():
    raw_data, ref = load_data()
    data = adjust_ref(raw_data, ref)
    res = fit_dg(data["Wavelength"], data["Intensity"])
    plot(res, raw_data, ref, data, "Wavelength", "Intensity")
    input("Does this look correct? [Y/n] ")
