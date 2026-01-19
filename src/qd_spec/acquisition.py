# Copyright 2024 StellarNet, Inc.
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

# For Windows ONLY: Must be run in administrator mode
# Only need to run it one time after switch back from the SpectraWiz. Or manually run InstallDriver.exe located in stellarnet_driverLis/windows_only
# sn.installDeviceDriver()
# This resturn a Version number of compilation date of driver
version = sn.version()
print(version)
# Device parameters to set
inttime = 50
scansavg = 1
smooth = 0
xtiming = 3
# init Spectrometer - Get BOTH spectrometer and wavelength
spectrometer, wav = sn.array_get_spec(0)  # 0 for first channel and 1 for second channel , up to 127 spectrometers
"""
# Equivalent to get spectrometer and wav separately:
spectrometer = sn.array_get_spec_only(0)
wav = sn.getSpectrum_X(spectrometer)
"""
print(spectrometer)
sn.ext_trig(spectrometer, True)
# Get device ID
deviceID = sn.getDeviceId(spectrometer)
print("\nMy device ID: ", deviceID)
# Get current device parameter
currentParam = sn.getDeviceParam(spectrometer)
# Call to Enable or Disable External Trigger to by default is Disbale=False -> with timeout
# Enable or Disable Ext Trigger by Passing True or False, If pass True than Timeout function will be disable, so user can also use this function as timeout enable/disbale
sn.ext_trig(spectrometer, True)
# Only call this function on first call to get spectrum or when you want to change device setting.
# -- Set last parameter to 'True' throw away the first spectrum data because the data may not be true for its inttime after the update.
# -- Set to 'False' if you want to throw away the first data, however your next spectrum data might not be valid.
sn.setParam(spectrometer, inttime, scansavg, smooth, xtiming, True)
# Get spectrometer data - Get BOTH X and Y in single return
first_data = sn.array_spectrum(spectrometer, wav)  # get specturm for the first time
"""
# Get Y value ONLY :
first_data = sn.getSpectrum_Y(spectrometer)
"""
print("First data:", first_data)
# ==============================================
# Burst FIFO mode: Not recommended with high integration time.
# burst_data_2 = sn.getBurstFifo_Y(spectrometer)
# ==============================================
# Release the spectrometer before ends the program
sn.reset(spectrometer)
# For Windows ONLY: Must be run in administrator mode
# sn.uninstallDeviceDriver()
