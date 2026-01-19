from .fitter import fit_dg
from .loader import load_data
from .plotter import plot


def main():
    raw_data, ref, data = load_data()
    res = fit_dg(data["Wavelength"], data["Intensity"])
    plot(res, raw_data, ref, data, "Wavelength", "Intensity")
