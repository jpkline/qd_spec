import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .fitter import fit_dg

FG_COLOR = "#F2F2F2"
BG_COLOR = "#0C0C0C"


def main():
    matplotlib.use("module://matplotlib-sixel-backend")
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "savefig.facecolor": BG_COLOR,
            #
            "text.color": FG_COLOR,
            "axes.labelcolor": FG_COLOR,
            "axes.titlecolor": FG_COLOR,
            "xtick.color": FG_COLOR,
            "ytick.color": FG_COLOR,
            "axes.edgecolor": FG_COLOR,
        }
    )
    plt.rcParams["lines.dash_capstyle"] = "round"
    plt.rcParams["lines.solid_capstyle"] = "round"

    file = pd.read_csv(r"D:\qd_anal\Raw_Spectra\OldSetup\new_emis.csv", names=["nm", "Abs"])
    fit_dg(file["nm"], file["Abs"])
