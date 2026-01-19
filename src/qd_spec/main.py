from .fitter import fit_dg
from .loader import adjust_ref, load_data
from .plotter import plot


def main():
    raw_data, ref = load_data()
    data = adjust_ref(raw_data, ref)
    res = fit_dg(data["Wavelength"], data["Intensity"])
    plot(res, raw_data, ref, data, "Wavelength", "Intensity")
    input("Does this look correct? [Y/n] ")
