
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import EnergyFilter, LowPassFilter 

def is_white_noise(x: np.array, alpha: float = 0.05) -> bool:
    """
    Check if the input signal is white noise or not
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    p_value = acorr_ljungbox(x, lags=1)[1]
    return p_value > alpha