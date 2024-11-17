import os
import torchaudio
import pandas as pd

# Define paths
audio_path = "/export/c09/lavanya/seame_data/data/interview/audio"
transcript_path = "/export/c09/lavanya/seame_data/data/interview/transcript/phaseII"
output_path = "/export/c09/lavanya/languageIdentification/seame/segmentInt"
report_file = "/export/c09/lavanya/languageIdentification/seame/reportInt.csv"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Target sample rate
target_sample_rate = 16000

# DataFrame to track processing results
report_data = []

# Function to save audio segments with resampling
def save_audio_segment(audio, start_sample, end_sample, output_file, original_sample_rate, target_sample_rate):
    # Slice the audio tensor based on start and end samples
    segment = audio[:, start_sample:end_sample]

    # Resample the segment to 16 kHz if needed
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        segment = resampler(segment)

    # Save the segment as a new audio file
    torchaudio.save(output_file, segment, target_sample_rate)
    print(f"Saved segment: {output_file}")

# Process each transcript file
txt_files = [f for f in os.listdir(transcript_path) if f.endswith('.txt')]
print(f"Number of .txt files: {len(txt_files)}")

for txt_file in txt_files:
    transcript_file_path = os.path.join(transcript_path, txt_file)
    audio_file_name = os.path.splitext(txt_file)[0] + ".flac"  # Assuming corresponding audio is a .flac file
    audio_file_path = os.path.join(audio_path, audio_file_name)

    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found for {txt_file}, skipping...")
        report_data.append({
            "language_tag": None,
            "transcript_text": None,
            "transcript_file": txt_file,
            "audio_file": audio_file_name,
            "segment_stored": None,
            "status": "Missing Audio"
        })
        continue

    # Load the audio file
    audio, sample_rate = torchaudio.load(audio_file_path)
    print(f"Processing {audio_file_name}, Original Sample Rate: {sample_rate}")

    # Read the transcript file
    with open(transcript_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3:
                start_time = int(parts[1])  # Start time in milliseconds
                end_time = int(parts[2])  # End time in milliseconds
                language_code = parts[3]  # Language code (EN, ZH, CS)

                # Skip segments with 'CS'
                if language_code == "CS":
                    continue

                # Append language-specific suffix
                if language_code == "EN":
                    language_tag = "English"
                elif language_code == "ZH":
                    language_tag = "Mandarin"
                else:
                    continue

                # Convert milliseconds to sample index
                start_sample = int((start_time / 1000) * sample_rate)
                end_sample = int((end_time / 1000) * sample_rate)

                # Generate output filename
                output_filename = f"{os.path.splitext(txt_file)[0]}_{start_time}_{end_time}_{language_tag}.wav"
                output_filepath = os.path.join(output_path, output_filename)

                # Save the audio segment with resampling
                save_audio_segment(audio, start_sample, end_sample, output_filepath, sample_rate, target_sample_rate)

                # Log segment information along with transcript text
                report_data.append({
                    "language_tag": language_tag,
                    "transcript_text": parts[4],  # Assuming the transcript text is in the 5th column
                    "transcript_file": txt_file,
                    "audio_file": audio_file_name,
                    "segment_stored": output_filepath,
                    "status": "Processed"
                })

# Save the report to a CSV file
report_df = pd.DataFrame(report_data)

# Reorder columns to put language_tag and transcript_text first
report_df = report_df[["language_tag", "transcript_text", "transcript_file", "audio_file", "segment_stored", "status"]]

# Save the final report CSV
report_df.to_csv(report_file, index=False)
print(f"Processing report saved to {report_file}")
