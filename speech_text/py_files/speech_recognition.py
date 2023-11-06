import torch
import json
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_pretrained_model_and_processor():
    """
    Load a pre-trained Wav2Vec 2.0 model and its processor.

    Returns:
    - processor: The Wav2Vec2Processor associated with the model.
    - model: The Wav2Vec2ForCTC model ready for transcription prediction.
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

def speech_file_to_array_fn(audio_filepath):
    """
    Convert an audio file to an array format expected by the model.

    Parameters:
    - audio_filepath (str): Path to the audio file to convert.

    Returns:
    - speech_array: The audio data as an array.
    - sampling_rate: The sampling rate of the audio file.
    """
    speech_array, sampling_rate = sf.read(audio_filepath)
    return speech_array, sampling_rate

def predict_transcription(audio_filepath, processor, model):
    """
    Predict the transcription for an audio file using a pre-trained model.

    Parameters:
    - audio_filepath (str): Path to the audio file for transcription.
    - processor: The processor associated with the Wav2Vec 2.0 model.
    - model: The pre-trained Wav2Vec 2.0 model.

    Returns:
    - transcription (str): The predicted transcription of the audio file.
    """
    # Read and process the audio file
    speech, sampling_rate = speech_file_to_array_fn(audio_filepath)
    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Ensure 'attention_mask' is present
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones(inputs.input_values.shape, dtype=torch.long)

    # Predict and decode the transcription
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Main script to load the model and process the dataset
if __name__ == "__main__":
    # Load the pre-trained model and processor
    processor, model = load_pretrained_model_and_processor()

    # Load the dataset from the JSON file
    with open('/home/posiden/Documents/GitHub/IDC409/speech_text/all_Data_wav/speech_dataset.json', 'r') as f:
        data = json.load(f)

    # Process each record in the dataset
    for record in data:
        audio_filepath = record['audio_filepath']
        true_transcription = record['transcription']
        predicted_transcription = predict_transcription(audio_filepath, processor, model)

        print(f"True: {true_transcription}")
        print(f"Pred: {predicted_transcription}")
        print("-----")
