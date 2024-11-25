import os
import pandas as pd
import numpy as np

from video_utils import extract_frames, preprocess_all_frames, get_video_info
from phys_data_utils import resample_signals


def get_subject_folders(data_root):
    """
    Get a list of subject folders in the dataset.
    
    Parameters:
        data_root (str): Path to the root directory of the dataset.
    
    Returns:
        List[str]: List of paths to subject folders.
    """
    subject_folders = []
    for folder_name in os.listdir(data_root):
        subject_path = os.path.join(data_root, folder_name)
        if os.path.isdir(subject_path) and folder_name.startswith('s'):
            subject_folders.append(subject_path)
    return subject_folders

def get_trial_files(subject_folder, subject_id):
    """
    Get the file paths for BVP, EDA, and video files for all trials of a subject.
    
    Parameters:
        subject_folder (str): Path to the subject's folder.
        subject_id (str): Subject identifier (e.g., 's1').
    
    Returns:
        List[dict]: List of dictionaries containing file paths for each trial.
    """
    trials = ['T1', 'T2', 'T3']
    trial_files = []
    for trial in trials:
        bvp_file = os.path.join(subject_folder, f"bvp_{subject_id}_{trial}.csv")
        eda_file = os.path.join(subject_folder, f"eda_{subject_id}_{trial}.csv")
        video_file = os.path.join(subject_folder, f"vid_{subject_id}_{trial}.avi")
        if os.path.exists(bvp_file) and os.path.exists(eda_file) and os.path.exists(video_file):
            trial_files.append({
                'trial': trial,
                'bvp_file': bvp_file,
                'eda_file': eda_file,
                'video_file': video_file
            })
    return trial_files

def process_video(video_file, subject_id, trial, desired_fps):
    """
    Extract and preprocess frames from a video file.
    
    Parameters:
        video_file (str): Path to the video file.
        subject_id (str): Subject identifier.
        trial (str): Trial identifier.
        desired_fps (int): Number of frames to extract per second.
    
    Returns:
        List[torch.Tensor]: List of preprocessed frame tensors.
    """
    output_folder = f"processed_frames/{subject_id}_{trial}"
    extract_frames(video_file, output_folder, frames_per_second=desired_fps)
    preprocessed_frames = preprocess_all_frames(output_folder, output_folder)
    return preprocessed_frames


def read_physiological_data(bvp_file, eda_file):
    """
    Read BVP and EDA data from CSV files.
    
    Parameters:
        bvp_file (str): Path to the BVP CSV file.
        eda_file (str): Path to the EDA CSV file.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: BVP and EDA data arrays.
    """
    bvp_data = pd.read_csv(bvp_file, header=None).iloc[:, 0].values
    eda_data = pd.read_csv(eda_file, header=None).iloc[:, 0].values
    return bvp_data, eda_data


def synchronize_data(bvp_data, eda_data, video_length, num_frames, bvp_sampling_rate, eda_sampling_rate):
    """
    Resample BVP and EDA data to match the number of video frames.
    
    Parameters:
        bvp_data (np.ndarray): Original BVP data.
        eda_data (np.ndarray): Original EDA data.
        video_length (float): Duration of the video in seconds.
        num_frames (int): Number of frames extracted from the video.
        bvp_sampling_rate (float): Sampling rate of BVP data.
        eda_sampling_rate (float): Sampling rate of EDA data.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled BVP and EDA data arrays.
    """
    desired_fps = num_frames / video_length
    new_time, eda_resampled, bvp_resampled = resample_signals(
        eda_data, bvp_data, video_length, desired_fps,
        eda_sampling_rate=eda_sampling_rate, bvp_sampling_rate=bvp_sampling_rate
    )
    return eda_resampled, bvp_resampled


def create_dataset(preprocessed_frames, bvp_resampled, eda_resampled):
    """
    Create a dataset from preprocessed frames and resampled physiological data.
    
    Parameters:
        preprocessed_frames (List[torch.Tensor]): List of preprocessed frame tensors.
        bvp_resampled (np.ndarray): Resampled BVP data.
        eda_resampled (np.ndarray): Resampled EDA data.
    
    Returns:
        List[dict]: List of samples, each containing 'frame', 'bvp', and 'eda'.
    """
    dataset = []
    num_samples = len(preprocessed_frames)
    for i in range(num_samples):
        sample = {
            'frame': preprocessed_frames[i],
            'bvp': bvp_resampled[i],
            'eda': eda_resampled[i]
        }
        dataset.append(sample)
    return dataset



def process_all_subjects(data_root, desired_fps, bvp_sampling_rate=256, eda_sampling_rate=256):
    """
    Process all subjects and trials to create a complete dataset.
    
    Parameters:
        data_root (str): Path to the root directory of the dataset.
        desired_fps (int): Frames per second for video processing.
        bvp_sampling_rate (float): Sampling rate of BVP data.
        eda_sampling_rate (float): Sampling rate of EDA data.
    
    Returns:
        List[dict]: Combined dataset from all subjects and trials.
    """
    subject_folders = get_subject_folders(data_root)
    all_data = []
    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        trials = get_trial_files(subject_folder, subject_id)
        for trial in trials:
            try:
                print(f"Processing {subject_id} {trial['trial']}")
                # Process video
                preprocessed_frames = process_video(trial['video_file'], subject_id, trial['trial'], desired_fps)
                
                if not preprocessed_frames:
                    print(f"No frames processed for {subject_id} {trial['trial']}. Skipping.")
                    continue
                
                # Get video info
                video_info = get_video_info(trial['video_file'])
                video_length = video_info['duration']
                num_frames = len(preprocessed_frames)
                
                # Read physiological data
                bvp_data, eda_data = read_physiological_data(trial['bvp_file'], trial['eda_file'])
                
                # Synchronize data
                eda_resampled, bvp_resampled = synchronize_data(
                    bvp_data, eda_data, video_length, num_frames, bvp_sampling_rate, eda_sampling_rate
                )
                
                # Ensure data lengths match
                min_length = min(len(preprocessed_frames), len(eda_resampled), len(bvp_resampled))
                preprocessed_frames = preprocessed_frames[:min_length]
                eda_resampled = eda_resampled[:min_length]
                bvp_resampled = bvp_resampled[:min_length]
                
                # Create dataset
                dataset = create_dataset(preprocessed_frames, bvp_resampled, eda_resampled)
                
                # Append to all_data
                all_data.extend(dataset)
            except Exception as e:
                print(f"Error processing {subject_id} {trial['trial']}: {e}")
    return all_data



