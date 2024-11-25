# video_utils.py

import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

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

def get_video_length(video_path):
    """
    Get the length of a video in seconds.

    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        float: Length of the video in seconds.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frames per second (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the duration in seconds
    if fps > 0:
        duration = frame_count / fps
    else:
        print("Error: FPS value is zero, cannot calculate duration.")
        duration = None
    
    # Release the video capture object
    cap.release()
    
    return duration

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
            
            # Print every 100 saved frames
            if saved_frame_count % 100 == 0:
                print(f"{saved_frame_count} frames saved")
        
        # Increment the frame count
        frame_count += 1

    # Print final count
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

def preprocess_all_frames(input_folder, output_folder=None):
    """
    Preprocess all frames from a folder.

    Parameters:
        input_folder (str): Folder containing extracted frames.
        output_folder (str, optional): Folder to save processed frames. If None, results are not saved.

    Returns:
        list[torch.Tensor]: List of preprocessed frame tensors.
    """
    # Define allowed image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
    ])
    
    # Ensure output directory exists
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    preprocessed_frames = []

    # Filter and process only valid image files
    for frame_name in sorted(os.listdir(input_folder)):
        frame_path = os.path.join(input_folder, frame_name)
        
        # Check if it's a file and has a valid extension
        if os.path.isfile(frame_path) and os.path.splitext(frame_name)[1].lower() in valid_extensions:
            try:
                # Open the image
                img = Image.open(frame_path)
                
                # Apply preprocessing
                processed_frame = transform(img)
                preprocessed_frames.append(processed_frame)
                
                # Save processed image if output_folder is provided
                if output_folder:
                    output_frame_path = os.path.join(output_folder, frame_name)
                    processed_img = transforms.ToPILImage()(processed_frame)  # Convert back to PIL image
                    processed_img.save(output_frame_path)
            except Exception as e:
                print(f"Failed to process {frame_name}: {e}")
    
    print(f"Processed {len(preprocessed_frames)} frames.")
    return preprocessed_frames


