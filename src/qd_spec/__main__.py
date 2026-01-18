import matplotlib
import pandas as pd

from .fitter import fit_dg

matplotlib.use("module://matplotlib-sixel-backend")

# plt.style.use("dark_background")

file = pd.read_csv(r"D:\qd_anal\Raw_Spectra\OldSetup\new_emis.csv", names=["nm", "Abs"])
fit_dg(file["nm"], file["Abs"])
