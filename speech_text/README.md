# Speech Recognition with Wav2Vec 2.0

This repository contains a set of functions designed to facilitate the processing of speech data and to perform speech recognition using the pre-trained Wav2Vec 2.0 model.

## Project Structure

The project includes the following key scripts:

- `copy_flac_files.py`: Consolidates all FLAC files from various subdirectories into a single directory.
- `convert_to_wav.py`: Converts all FLAC audio files in a directory to WAV format, suitable for processing by the Wav2Vec 2.0 model.
- `generate_dataset_json.py`: Parses transcription files to create a JSON dataset that maps audio file paths to their corresponding transcriptions.
- `speech_recognition.py`: Loads the pre-trained Wav2Vec 2.0 model and predicts transcriptions for audio files in the dataset.

## Setup

To run the scripts, you will need to install the required Python packages. Create a virtual environment and install the dependencies as follows:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The requirements.txt file should list the following libraries:

- torch
- transformers
- soundfile
- pydub

Make sure you have the `ffmpeg` library installed on your system as it is required by `pydub` for audio format conversion.

## Usage

1- **Copy FLAC Files**: Run `copy_flac_files.py` to consolidate your FLAC files. 
```python
python copy_flac_files.py
```

2- **Convert FLAC to WAV**: Execute `convert_to_wav.py` to convert all FLAC files to WAV format.
```python
python convert_to_wav.py
```
3- **Generate Dataset JSON**: Use `generate_dataset_json.py` to create a JSON file that contains the mapping of audio file paths to transcriptions.
```python
python generate_dataset_json.py
```

4- **Speech Recognition**: With `speech_recognition.py`, you can predict transcriptions for the audio files using the pre-trained Wav2Vec 2.0 model.
```python
python speech_recognition.py
```
## Output
After running the speech recognition script, you will get the predicted transcription of each audio file printed to the console along with its true transcription for comparison.

## Customization
You can adjust the paths and other configurations by modifying the respective variables in each script to suit your directory structure and data.

## Contributing
Contributions to improve the scripts or extend their functionality are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## Dataset

The speech recognition model in this project has been trained and tested using the [LibriSpeech ASR corpus](https://www.openslr.org/12/). The LibriSpeech dataset is a collection of approximately 1000 hours of 16kHz read English speech, derived from audiobooks from the LibriVox project, and has been carefully segmented and aligned.

### Dataset Structure

The LibriSpeech dataset is split into several subsets to facilitate different training and evaluation tasks:

- `train-clean-100`, `train-clean-360`, `train-other-500`: Training datasets of varying sizes and acoustic conditions.
- `dev-clean`, `dev-other`: Development datasets used for model validation.
- `test-clean`, `test-other`: Testing datasets used to evaluate the final model performance.

For the purpose of this project, we assume the dataset has been pre-processed and organized into a format where each audio file has a corresponding transcription.

### Attribution

When using the LibriSpeech dataset, please include the following citation:

*Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5206-5210). IEEE.*


## Acknowledgments

We would like to thank the creators of the LibriSpeech dataset for providing a high-quality, freely available resource for the speech recognition community.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

