import numpy as np
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt



def resample_signals(eda_data, bvp_data, video_length, desired_fps, eda_sampling_rate=None, bvp_sampling_rate=None):
    # Original number of samples
    num_eda_samples = len(eda_data)
    num_bvp_samples = len(bvp_data)

    # Use provided sampling rates if available
    if eda_sampling_rate is None:
        eda_sampling_rate = num_eda_samples / video_length

    if bvp_sampling_rate is None:
        bvp_sampling_rate = num_bvp_samples / video_length

    # Generate original timestamps
    eda_time = np.arange(0, num_eda_samples) / eda_sampling_rate
    bvp_time = np.arange(0, num_bvp_samples) / bvp_sampling_rate

    # Adjust timestamps to match video length if necessary
    eda_time = eda_time[:num_eda_samples]
    bvp_time = bvp_time[:num_bvp_samples]

    # New timestamps at desired FPS
    num_frames = int(video_length * desired_fps)
    new_time = np.linspace(0, video_length, num=num_frames)

    # Interpolate EDA data
    eda_interpolator = interp1d(eda_time, eda_data, kind='linear', fill_value="extrapolate")
    eda_resampled = eda_interpolator(new_time)

    # Interpolate BVP data
    bvp_interpolator = interp1d(bvp_time, bvp_data, kind='linear', fill_value="extrapolate")
    bvp_resampled = bvp_interpolator(new_time)

    return new_time, eda_resampled, bvp_resampled



def plot_resampled_signals(new_time, eda_resampled, bvp_resampled):
    """
    Plots the resampled EDA and BVP signals over time.

    Parameters:
        new_time (array): Array of timestamps for resampled data.
        eda_resampled (array): Resampled EDA signal data.
        bvp_resampled (array): Resampled BVP signal data.
    """
    # Create a figure with two subplots
    plt.figure(figsize=(14, 8))

    # Plot EDA Signal
    plt.subplot(2, 1, 1)  # First subplot
    plt.plot(new_time, eda_resampled, label="EDA Signal", color='blue')
    plt.title("Resampled EDA Signal Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("EDA Signal Value")
    plt.grid(True)
    plt.legend()

    # Plot BVP Signal
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(new_time, bvp_resampled, label="BVP Signal", color='red')
    plt.title("Resampled BVP Signal Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("BVP Signal Value")
    plt.grid(True)
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()