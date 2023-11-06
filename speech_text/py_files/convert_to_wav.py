import os
from pydub import AudioSegment

def convert_flac_to_wav(flac_file_path, wav_file_path):
    """
    Convert an audio file from FLAC format to WAV format.

    Parameters:
    - flac_file_path (str): The file path of the source FLAC file.
    - wav_file_path (str): The file path where the WAV file will be saved.
    """
    # Load the FLAC file
    audio = AudioSegment.from_file(flac_file_path, 'flac')
    # Export the audio in WAV format
    audio.export(wav_file_path, format='wav')

# Directory containing the original FLAC files
source_dir = 'all_Data'
# Directory where the WAV files will be saved
target_dir = 'all_Data_wav'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Iterate over all files in the source directory
for file in os.listdir(source_dir):
    if file.endswith('.flac'):
        # Full path to the source FLAC file
        flac_file_path = os.path.join(source_dir, file)
        # Full path to the target WAV file (change extension to .wav)
        wav_file_path = os.path.join(target_dir, file.replace('.flac', '.wav'))
        # Convert the FLAC file to WAV format and save it
        convert_flac_to_wav(flac_file_path, wav_file_path)
        print(f"Converted {file} to WAV format.")

print("Finished converting all FLAC files to WAV format.")
