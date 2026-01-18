import pandas as pd

from .fitter import fit_dg
from .plotter import plot


def main():
    file = pd.read_csv(r"D:\qd_anal\Raw_Spectra\OldSetup\new_emis.csv", names=["nm", "Abs"])
    res = fit_dg(file["nm"], file["Abs"])
    plot(res, file["nm"], file["Abs"])
