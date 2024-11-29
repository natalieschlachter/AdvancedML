def authenticate_drive():
    """
    Authenticates and returns the Google Drive instance.
    """
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("/work/Project/client_secrets.json")  # Make sure this file is in the same folder as the script
    gauth.CommandLineAuth()  # Use command line authentication instead of local server
    return GoogleDrive(gauth)


def get_or_create_folder(folder_name, drive, parent_folder_id=None):
    """
    Gets or creates a folder in Google Drive.

    Parameters:
    - folder_name (str): Name of the folder.
    - drive (GoogleDrive): Authenticated GoogleDrive instance.
    - parent_folder_id (str): Parent folder ID (optional).

    Returns:
    - str: Folder ID of the found or created folder.
    """
    query = f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"

    folder_list = drive.ListFile({'q': query}).GetList()
    
    if folder_list:
        return folder_list[0]['id']
    else:
        folder_metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': parent_folder_id}] if parent_folder_id else []
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        print(f"Created folder '{folder_name}' with ID: {folder['id']}")
        return folder['id']


def upload_file_to_drive(file_path, drive, parent_folder_id=None, counter=None, lock=None):
    """
    Uploads a file to Google Drive.

    Parameters:
    - file_path (str): Path to the local file.
    - drive (GoogleDrive): Authenticated GoogleDrive instance.
    - parent_folder_id (str): Parent folder ID (optional).
    - counter (dict): Dictionary to keep track of the number of uploaded files.
    - lock (Lock): Lock to ensure thread-safe operations on the counter.
    """
    file_name = os.path.basename(file_path)
    gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': parent_folder_id}] if parent_folder_id else []})
    gfile.SetContentFile(file_path)
    gfile.Upload()
    
    with lock:
        counter['count'] += 1
        print(f"Uploaded {file_name} to Google Drive ({counter['count']} files uploaded so far)")


def upload_folder_to_drive(folder_path, drive, parent_folder_id=None, max_workers=5):
    """
    Uploads all files from a given folder to Google Drive using multi-threading.

    Parameters:
    - folder_path (str): Path to the local folder.
    - drive (GoogleDrive): Authenticated GoogleDrive instance.
    - parent_folder_id (str): Parent folder ID (optional).
    - max_workers (int): Max threads for concurrent uploads.
    """
    file_paths = [
        os.path.join(root, file_name)
        for root, _, files in os.walk(folder_path)
        for file_name in files
    ]

    counter = {'count': 0}
    lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in file_paths:
            executor.submit(upload_file_to_drive, file_path, drive, parent_folder_id, counter, lock)


if __name__ == "__main__":
    dataset_dir = '/work/Project/FinalData'  # Change this to your dataset location

    # Authenticate Google Drive
    drive = authenticate_drive()

    # Get or create the "Dataset" folder in Google Drive
    dataset_folder_id = get_or_create_folder("Dataset", drive)

    # Upload the entire folder to Google Drive
    upload_folder_to_drive(dataset_dir, drive, parent_folder_id=dataset_folder_id)


