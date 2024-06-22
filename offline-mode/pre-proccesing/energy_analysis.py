
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from pyshm.dataShaper import shaper, MeanCentering
from pyshm.scaling import Ztranform , Normalization
from pyshm.filters import EnergyFilter, LowPassFilter 
from statsmodels.graphics.tsaplots import plot_acf


def is_white_noise(x:np.array, fs:int =100):
    """
    This function analyzes a signal to see if it has characteristics of white noise.

    Args:
        signal: A NumPy array representing the signal.
        fs: The sampling rate of the signal (default: 44100 Hz).

    Returns:
        A dictionary containing analysis results:
            - is_white: A boolean indicating if the signal has properties of white noise (not guaranteed).
            - psd_flat: A boolean indicating if the power spectral density is relatively flat.
            - autocorr_spike: A boolean indicating if the autocorrelation has a spike at zero lag.
    """

    # Calculate the Power Spectral Density (PSD)
    fft = np.fft.fft(x)
    psd = np.abs(fft) ** 2 / len(x)
    psd_db = 20 * np.log10(psd / np.max(psd))  # Convert to decibels

    # Analyze PSD flatness
    psd_std = np.std(psd_db)
    psd_flat = psd_std < 5  # Threshold for a relatively flat PSD

    # Calculate autocorrelation
    autocorr = np.correlate(x, x, mode='full')
    autocorr_spike = np.abs(autocorr[len(x) // 2]) > 0.9 * np.max(autocorr)  # Check for significant spike at zero lag

    # Make a basic judgement based on both characteristics
    is_white = psd_flat and autocorr_spike
    analysis = {
        "is_white": is_white,
        "psd_flat": psd_flat,
        "autocorr_spike": autocorr_spike
    }
    return analysis, autocorr, psd_db


def from_raw_to_energy(x, VERBOSE = False):
    
    sequence_length = 100
    windpw_stride = 100
    data_range = (-1, 1)
    
    data_shaper = shaper(sequence_len = sequence_length, stride = windpw_stride)
    MeanCenter = MeanCentering(axis = 0)
    ztransform = Ztranform()
    minmax_norm = Normalization(feature_range= data_range, clip = False)
    energy_filtering = EnergyFilter(threshold = 0, axis = 1) 

    x_tr_center = MeanCenter(x)

    sns.set_style(style="whitegrid")


    
    # Shape the data 
    x_tr_shaped = data_shaper(x.reshape( -1, 1))
    

    if VERBOSE == True:  
        print(f"Train Shape = {x_tr_shaped.shape}")
        

    x_tr_scaled = ztransform(x_tr_shaped, input_type = 'train')
    

    # if VERBOSE == True:  
    #     print(f"Scaled Train Shape = {x_tr_scaled.shape}")
         

    x_tr_norm = minmax_norm(x_tr_scaled, input_type = 'train')
    
    x_energy = energy_filtering(x_tr_shaped)

    energies = energy_filtering._fit(x_tr_shaped)

    return x_energy, energies

def main():
    # Example usage (replace 'your_signal' with your actual data)
    path = "../../data/"
    df_ = pd.read_feather(f"{path}/exp_3.feather")
    df_x = df_.x.values # [7_000:]
    df_time = df_.ts.values  # [7_000:]
    print(df_.head())
    time_mask = (df_time > pd.to_datetime( "2022-06-23 11:44:00.000")) &  (df_time < pd.to_datetime( "2022-06-23 11:45:20.000")) 
    print(time_mask)
    df_clip = df_[time_mask]


    plot_acf(df_clip.x.values, lags = 5)
    plt.show()
    exit()
    # plot the signal 
    plt.figure(figsize=(12, 6))
    # plt.plot(df_time, df_x)
    plt.plot(df_clip.ts.values, df_clip.x.values)
    
    data, energies = from_raw_to_energy(df_clip.x.values, VERBOSE = True)
    plt.figure(figsize=(12, 6))
    plt.plot(energies)
    plt.show()

    print(data.shape)
    # exit()
    
    analysis, autocorr, psd = is_white_noise(data[1, :])

    print("Is the signal white noise (indication only):", analysis["is_white"])
    print("Power spectral density flat:", analysis["psd_flat"])
    print("Autocorrelation has spike at zero lag:", analysis["autocorr_spike"])

    # Note: This is a basic analysis and may not be conclusive for all cases.

    # Plot the PSD and autocorrelation for visual inspection
    plt.figure(figsize=(12, 6))
    plt.plot(psd, label="Power Spectral Density (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Power Spectral Density")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plt.plot(autocorr, label="Autocorrelation")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("")
    plt.title("Autocorrelation")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()