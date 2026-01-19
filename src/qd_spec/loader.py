import pathlib

import pandas as pd

# For now, just try one sample
FNAME = "011726_Sample_070324a_b.SSM"
REF_FNAME = "011726_Reference_Tolulene_b.SSM"

DIRECTORY = r"C:\Users\jpkli\OneDrive - Benedictine College\LSC Research\Data\Laser_Spectra"


def load_data():
    raw_data = pd.read_csv(
        pathlib.Path(DIRECTORY) / FNAME, skiprows=2, delimiter=r"\s+", names=["Wavelength", "Intensity"]
    )
    ref = pd.read_csv(
        pathlib.Path(DIRECTORY) / REF_FNAME, skiprows=2, delimiter=r"\s+", names=["Wavelength", "Intensity"]
    )
    return raw_data, ref


def adjust_ref(raw_data, ref):
    data = raw_data.copy()
    data["Intensity"] -= ref["Intensity"]
    return data
