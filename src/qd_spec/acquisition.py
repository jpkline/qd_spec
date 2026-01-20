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
