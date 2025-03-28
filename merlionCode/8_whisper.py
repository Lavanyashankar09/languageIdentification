import os
import whisper
import csv
from tqdm import tqdm

# Load Whisper model (choose: "tiny", "base", "small", "medium", "large")
model = whisper.load_model("large")

# Folder containing audio files
AUDIO_FOLDER = "/export/c09/lavanya/languageIdentification/merlion/segment/"
OUTPUT_CSV = "/export/c09/lavanya/languageIdentification/merlion/whisper_transcriptions.csv"

# Get all audio files in the folder (limit to first 5 files)
audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(('.mp3', '.wav', '.m4a'))]
audio_files = audio_files  # Process only the first 5 files

# Open the CSV file in write mode
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Audio File', 'Transcription (Uppercase)', 'Transcription (List of Words)'])

    # Process each file
    for file in tqdm(audio_files, desc="Transcribing Audio"):
        audio_path = os.path.join(AUDIO_FOLDER, file)

        # Transcribe
        result = model.transcribe(audio_path)

        # Convert transcription to uppercase
        transcription_upper = result["text"].upper()

        # Split transcription into a list of words
        transcription_list = transcription_upper.split()

        # Write the audio file name, uppercase transcription, and list of words to the CSV
        writer.writerow([file, transcription_upper, str(transcription_list)])

        print(f"Transcribed: {file} → {OUTPUT_CSV}")
