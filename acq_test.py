import pandas as pd
from stellarnet_driverLibs import stellarnet_driver3 as sn

inttime = 50
scansavg = 1
smooth = 0
xtiming = 3

spectrometer, wav = sn.array_get_spec(0)
sn.ext_trig(spectrometer, False)
sn.setParam(spectrometer, inttime, scansavg, smooth, xtiming, True)

df = pd.DataFrame(sn.array_spectrum(spectrometer, wav))
df.plot(kind="scatter", x=0, y=1)

sn.reset(spectrometer)
