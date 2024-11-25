# video_utils.py

import cv2
import os
import matplotlib.pyplot as plt

def open_video(video_path):
    """
    Opens a video file and returns the VideoCapture object.
    
    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        cap (cv2.VideoCapture): VideoCapture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap


def save_frames(cap, output_dir="frames", frame_limit=5):
    """
    Saves a specified number of frames as images.
    
    Parameters:
        cap (cv2.VideoCapture): VideoCapture object.
        output_dir (str): Directory where frames will be saved.
        frame_limit (int): Number of frames to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_num = 0

    while cap.isOpened() and frame_num < frame_limit:
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
            success = cv2.imwrite(frame_path, frame)
            if success:
                print(f"Saved frame to {frame_path}")
            frame_num += 1
        else:
            break

def display_frames(cap, frame_limit=3):
    """
    Displays a specified number of frames using Matplotlib.
    
    Parameters:
        cap (cv2.VideoCapture): VideoCapture object.
        frame_limit (int): Number of frames to display.
    """
    frame_num = 0
    while cap.isOpened() and frame_num < frame_limit:
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for Matplotlib display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame_rgb)
            plt.axis('off')
            plt.show()
            frame_num += 1
        else:
            break



def extract_frames(video_path, output_folder, frames_per_second):
    """
    Extracts frames from a video at a specified rate.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_folder (str): Directory where extracted frames will be saved.
    - frames_per_second (int): Number of frames to extract per second (default is 1).

    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame interval based on the desired extraction rate
    frame_interval = int(fps / frames_per_second)
    
    # Frame counter
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        
        # Break the loop if no frames are returned (end of video)
        if not ret:
            break
        
        # Save the frame if it's at the desired interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1
            print(f"Saved {output_path}")
        
        # Increment the frame count
        frame_count += 1

    print(f"Total frames saved: {saved_frame_count}")
    
    # Release the video capture object
    cap.release()




def get_video_info(video_path):
    """
    Retrieves and prints video information: frame count, FPS, and duration.

    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        dict: Dictionary containing 'frame_count', 'fps', and 'duration' in seconds.
    """
    cap = open_video(video_path)
    if not cap:
        return None

    # Get total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get frames per second (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate video duration
    if fps > 0:
        duration = frame_count / fps
    else:
        print("Error: FPS value is zero, cannot calculate duration.")
        duration = None

    cap.release()

    print(f"Total number of frames: {frame_count}")
    print(f"Frames per second (FPS): {fps}")
    print(f"Duration of the video: {duration} seconds")

    return {"frame_count": frame_count, "fps": fps, "duration": duration}



import pandas as pd

def calculate_bvp_rows(target_fps, video_duration):
    """
    Calculate the number of BVP rows after downsampling from 64 Hz to the target FPS.
    
    Parameters:
    - target_fps (float): Target FPS to downsample the BVP signal to.
    - video_duration (float): Duration of the video in seconds.
    
    Returns:
    - number_of_rows (int): The number of BVP rows after downsampling.
    """
    # Original BVP sampling rate is 64 Hz
    original_fps = 64  # BVP is sampled at 64 Hz
    
    # Calculate the number of BVP samples per frame (using video FPS)
    samples_per_frame = original_fps / 35.138  # BVP frequency (64 Hz) / Video FPS (35.138 FPS)
    
    # Downsampling BVP to target FPS
    samples_per_second_target = samples_per_frame * target_fps
    
    # Calculate the total number of rows for BVP
    number_of_rows = int(samples_per_second_target * video_duration)
    
    return number_of_rows


def calculate_eda_rows(target_fps, video_duration):
    """
    Calculate the number of EDA rows after upsampling from 4 Hz to the target FPS.
    
    Parameters:
    - target_fps (float): Target FPS to upsample the EDA signal to.
    - video_duration (float): Duration of the video in seconds.
    
    Returns:
    - number_of_rows (int): The number of EDA rows after upsampling.
    """
    # Original EDA sampling rate is 4 Hz
    original_fps = 4  # EDA is sampled at 4 Hz
    
    # Calculate the number of EDA samples per frame (using video FPS)
    samples_per_frame = original_fps / 35.138  # EDA frequency (4 Hz) / Video FPS (35.138 FPS)
    
    # Upsampling EDA to target FPS
    samples_per_second_target = samples_per_frame * target_fps
    
    # Calculate the total number of rows for EDA
    number_of_rows = int(samples_per_second_target * video_duration)
    
    return number_of_rows



import numpy as np

def downsample_bvp1(bvp, original_fps, target_fps, video_duration):
    """
    Downsample the BVP signal from original FPS to target FPS by averaging values.

    Parameters:
    - bvp (np.array or pd.Series): Original BVP signal (assumed to be at original_fps).
    - original_fps (float): The original FPS of the signal (e.g., 64 Hz for BVP).
    - target_fps (float): The target FPS to downsample the BVP signal to (e.g., 5 FPS).
    - video_duration (float): Duration of the video in seconds.

    Returns:
    - downsampled_bvp (np.array): Downsampled BVP signal.
    """
    # Calculate the number of samples per frame based on original FPS and video FPS
    samples_per_frame = original_fps / 35.138  # original_fps (e.g., 64 Hz) / video FPS (35.138 FPS)
    
    # Calculate the downsampling factor
    downsampling_factor = (samples_per_frame * target_fps)  # Multiply by target FPS and round to integer

    # Calculate the number of rows for the downsampled BVP
    num_samples = (target_fps * video_duration)

    # Downsample the BVP signal by averaging every `downsampling_factor` samples
    downsampled_bvp = []
    for i in range(0, len(bvp), downsampling_factor):  # Group by downsampling factor
        group = bvp[i:i + downsampling_factor]  # Group the samples into windows
        downsampled_bvp.append(np.mean(group))  # Average the values in the group

    return np.array(downsampled_bvp)
