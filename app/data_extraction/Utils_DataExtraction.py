
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to process a single folder
def process_folder(folder, dataset_dir, output_dir):
    """
    This function processes a single folder, randomly selects one video file,
    and extracts frames from the video based on the corresponding EDA signals.
    """
    folder_path = os.path.join(dataset_dir, folder)
    logging.info(f"Processing folder: {folder}")
    if not os.path.isdir(folder_path):
        return

    # Get all AVI files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
    if not video_files:
        logging.warning(f"No video files found in folder: {folder}. Skipping...")
        return

    # Randomly select one video
    selected_video = random.choice(video_files)
    video_path = os.path.join(folder_path, selected_video)
    logging.info(f"Selected video: {selected_video}")

    # Determine the corresponding EDA file
    base_name = os.path.splitext(selected_video)[0]
    T_part = base_name.split('_')[-1]  # Extract T1, T2, T3
    eda_file = f'eda_{folder}_{T_part}.csv'
    eda_path = os.path.join(folder_path, eda_file)
    logging.info(f"Looking for corresponding EDA file: {eda_file}")

    if not os.path.exists(eda_path):
        logging.warning(f"EDA file not found for {selected_video}. Skipping...")
        return

    # Load EDA signals
    logging.info(f"Loading EDA data from file: {eda_file}")
    try:
        eda_data = pd.read_csv(eda_path, header=None)
    except Exception as e:
        logging.error(f"Failed to read EDA file: {eda_file}. Error: {e}")
        return

    eda_sampling_rate = 4  # 4 Hz
    logging.info(f"EDA data loaded. Sampling rate: {eda_sampling_rate} Hz")

    # Open video and extract frames
    logging.info(f"Opening video file: {selected_video}")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {video_fps}, Total frames: {total_frames}, Video length: {video_length} seconds")

    # Generate new timestamps at EDA sampling rate
    num_frames = int(video_length * eda_sampling_rate)  # 180 * 4 = 720
    frame_timestamps = np.linspace(0, video_length, num=num_frames)  # Array of time points at which the frames should be extracted.

    # Adjust the last timestamp if it slightly exceeds the video length
    frame_timestamps[-1] = min(frame_timestamps[-1], video_length - 1.0 / video_fps)

    all_frames_data = []  # List to store frame filenames and corresponding EDA signals

    # Extract frames at the EDA sampling rate
    for i, timestamp in enumerate(frame_timestamps, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Set the video position in milliseconds
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"End of video reached or error reading the frame at timestamp {timestamp:.2f} seconds.")
            break

        logging.info(f"Extracting frame at time {timestamp:.2f} seconds (Signal frame count: {i})")
        # Save frame to output directory
        frame_filename = f"{folder}_{T_part}_frame_{i:04d}.jpg"
        output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(output_path, frame)
        logging.info(f"Saved frame to: {output_path}")

        # Store frame filename and corresponding EDA signal
        if i <= len(eda_data):
            all_frames_data.append((frame_filename, eda_data.iloc[i - 1, 0]))

    cap.release()
    logging.info(f"Finished processing video: {selected_video}. Total frames saved: {len(frame_timestamps)}")

    # Create a DataFrame with all frame data
    frames_df = pd.DataFrame(all_frames_data, columns=['Frame_Filename', 'EDA_Signal'])
    # Save the DataFrame to a CSV file
    frames_csv_path = os.path.join(output_dir, f"Data_{folder}_{T_part}.csv")
    frames_df.to_csv(frames_csv_path, index=False)
    logging.info(f"Dataset with frame filenames and EDA signals saved to: {frames_csv_path}")

# Function to process all folders in parallel
def process_folders_parallel(dataset_dir, output_dir):
    folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
    with ThreadPoolExecutor() as executor:
        executor.map(lambda folder: process_folder(folder, dataset_dir, output_dir), folders)


def analyze_dataset_and_visualize(dataset_path, csv_input_folder):
    jpg_count = 0
    csv_count = 0
    jpg_groups = defaultdict(list)
    video_groups = defaultdict(int)

    """
    This function allows to analyze and visualizes the extracted frames
    and check whether the output is as expected
    
    """
    # Traverse the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_count += 1
                parts = file.split('_')
                group = parts[0]  # Extract prefix (e.g., s1)
                video_number = parts[1]  # Extract video number (e.g., T1)

                jpg_groups[group].append((os.path.join(root, file), video_number))
                video_groups[video_number] += 1
            elif file.endswith('.csv'):
                csv_count += 1

    total_groups = len(jpg_groups)
    print(f"\nTotal number of patients: {total_groups}")
    print(f"\nTotal number of frames: {jpg_count}")
    print(f"\nNo. of frames/ No. of patients: {jpg_count / total_groups if total_groups > 0 else 'N/A'}")
    print(f"\nTotal number of .csv files: {csv_count}")

    print(f"\nNumber of frames per video number:")
    for video_number, count in sorted(video_groups.items()):
        print(f"  Video {video_number}: {count} .jpg files")

    # Sort groups in ascending order by the number after the 's'
    sorted_groups = sorted(jpg_groups.keys(), key=lambda x: int(x[1:]))

    # Per-group analysis
    for group in sorted_groups:
        jpg_files = jpg_groups[group]
        video_number = jpg_files[0][1]  # Extract the video number from the first file in the group

        # Build the path to the corresponding CSV file
        csv_file_path = os.path.join(csv_input_folder, group, f"eda_{group}_{video_number}.csv")

        # Compute the length of the CSV file (considering no header)
        if os.path.exists(csv_file_path):
            try:
                csv_length = len(pd.read_csv(csv_file_path, header=None))  # Read without header
            except Exception as e:
                print(f"Error reading {csv_file_path}: {e}")
                csv_length = "Error reading file"
        else:
            csv_length = "File not found"

        # Display group information
        print(f"\nPatient: {group}; video task: {video_number}")
        print(f"1. Number of frames extracted: {len(jpg_files)}")
        print(f"2. Eda signals file: {csv_file_path};  Number of signals: {csv_length}")

        # Display 10 random images
        print("3. Displaying 10 random images:")
        sample_files = random.sample(jpg_files, min(10, len(jpg_files)))  # Up to 10 random files

        plt.figure(figsize=(20, 10))
        for i, (file_path, _) in enumerate(sample_files):
            img = Image.open(file_path)
            plt.subplot(1, 10, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()