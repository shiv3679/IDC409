import os
from shutil import copy2

def unique_filename(file_path):
    """
    Generate a unique filename by appending a counter to the base name if the file already exists.

    Parameters:
    - file_path (str): The initial file path for which to check uniqueness.

    Returns:
    - str: A unique file path. If the initial file path does not exist, it is returned as is;
           otherwise, a counter is appended to the base name until a unique name is generated.
    """
    base, extension = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{counter}{extension}"
        counter += 1
    return file_path

# Base directory where the folders with FLAC files are located
base_dir = 'speech_data'
# Target directory where all FLAC files will be copied
target_data_dir = 'all_Data'
# Target subdirectory for all transcript files
target_transcripts_dir = os.path.join(target_data_dir, 'all_Transcripts')

# Create the target directories if they don't exist
os.makedirs(target_data_dir, exist_ok=True)
os.makedirs(target_transcripts_dir, exist_ok=True)

# Iterate over each subfolder and file in the base directory
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        # Full path to the source file
        file_path = os.path.join(subdir, file)
        # Determine the target directory based on file type
        if file.endswith('.flac'):
            target_dir = target_data_dir
        elif file.endswith('.txt'):
            target_dir = target_transcripts_dir
        else:
            # Skip any non-flac and non-txt files
            continue

        # Full path for the target file
        target_file_path = os.path.join(target_dir, file)
        
        # Ensure a unique filename in case of duplicates
        target_file_path = unique_filename(target_file_path)

        # Copy the file to the target directory
        copy2(file_path, target_file_path)
        print(f"Copied {file} to {target_dir}")

print("Finished copying all files.")
