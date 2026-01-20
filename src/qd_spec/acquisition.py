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

from stellarnet_driverLibs import stellarnet_driver3 as sn

INTTIME = 1000
SCANSAVG = 1  # FIXME: 10
SMOOTH = 0
XTIMING = 3
CHANNEL = 0


class SpectrometerAcquisition:
    def __init__(self):
        self.spectrometer = sn.array_get_spec_only(CHANNEL)
        sn.ext_trig(self.spectrometer, False)
        sn.setParam(self.spectrometer, INTTIME, SCANSAVG, SMOOTH, XTIMING, clear=True)

    def acquire_spectrum(self):
        return sn.getSpectrum_Y(self.spectrometer)

    def acquire_wavelengths(self):
        return sn.getSpectrum_X(self.spectrometer)

    def release(self):
        sn.reset(self.spectrometer)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
