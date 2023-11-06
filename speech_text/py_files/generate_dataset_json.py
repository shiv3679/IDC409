import json
import os

def create_dataset_json(transcripts_dir, audio_dir, output_json_path):
    """
    Create a JSON dataset file from transcripts and corresponding audio files.

    Parameters:
    - transcripts_dir (str): Path to the directory containing transcript files.
    - audio_dir (str): Path to the directory containing audio files.
    - output_json_path (str): Path where the output JSON file will be saved.
    """
    # This list will store all data points (audio file paths and their transcriptions)
    data_points = []

    # List all transcript files in the given directory with a specific extension
    transcript_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.trans.txt')]

    # Process each transcript file
    for transcript_file in transcript_files:
        with open(os.path.join(transcripts_dir, transcript_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Each line contains an audio file name and the transcription, separated by a space
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_filename = parts[0] + '.wav'
                    transcription = parts[1]
                    audio_filepath = os.path.join(audio_dir, audio_filename)
                    
                    # Check if the corresponding audio file exists
                    if os.path.isfile(audio_filepath):
                        # Add the audio file path and transcription to the dataset
                        data_points.append({
                            'audio_filepath': audio_filepath,
                            'transcription': transcription
                        })

    # Write the data points to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(data_points, json_file, indent=4)

    print(f"Created dataset with {len(data_points)} entries.")

# Paths to the required directories and output JSON file
transcripts_dir = '/home/posiden/Documents/GitHub/IDC409/speech_text/all_Data_wav/all_Transcripts'
audio_dir = '/home/posiden/Documents/GitHub/IDC409/speech_text/all_Data_wav'
output_json_path = '/home/posiden/Documents/GitHub/IDC409/speech_text/all_Data_wav/speech_dataset.json'

# Create the JSON dataset
create_dataset_json(transcripts_dir, audio_dir, output_json_path)
