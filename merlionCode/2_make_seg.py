import os
import pandas as pd
import librosa
import soundfile as sf
import pickle

class AudioSegmenter:
    def __init__(self, audio_folder, output_folder):
        self.audio_folder = audio_folder
        self.output_folder = output_folder

    def segment_audio_files(self, df):
        print("Starting audio segmentation...")
        segmented_data = []  # List to store segmented data
        for index, row in df.iterrows():
            print(f"Processing row {index + 1}/{len(df)}...")
            audio_file = os.path.join(self.audio_folder, row['audio_name'])
            output_file_prefix = os.path.splitext(row['audio_name'])[0]
            language_tag = row['language_tag']
            # Load audio and resample to 16 kHz
            try:
                audio, sr = librosa.load(audio_file, sr=16000)
                print(f"Loaded {audio_file} with sample rate {sr}.")
            except Exception as e:
                print(f"Failed to load {audio_file}. Reason: {e}")
                continue

            # Calculate segment boundaries based on 'start' and 'end' columns in DataFrame
            start_sample = int(row['start'] * sr / 1000)
            end_sample = int(row['end'] * sr / 1000)

            # Segment the audio
            segment = audio[start_sample:end_sample]

            # Save the segment
            output_file = os.path.join(self.output_folder, f"{output_file_prefix}_segment_{index}_{language_tag}.wav")
            try:
                sf.write(output_file, segment, sr, subtype='PCM_16')
                print(f"Saved segment to {output_file}.")
            except Exception as e:
                print(f"Failed to save segment to {output_file}. Reason: {e}")

            # Append to segmented_data list
            segmented_data.append({
                'segmented_audio': output_file,
                'language_tag': row['language_tag'],
                'overlap_diff_lang': row['overlap_diff_lang'],
                'length': row['length'],
                'utt_id': row['utt_id']
            })

        # Create DataFrame from segmented_data
        print("Creating DataFrame from segmented data...")
        segmented_df = pd.DataFrame(segmented_data)
        print("Segmentation complete.")
        return segmented_df

# Define the common path
common_path = "/export/c09/lavanya/merlion_data/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/"
#change this
output_path = "/export/c09/lavanya/languageIdentification/zinglish/large"

# Define specific paths using the common path
csv_path = os.path.join(common_path, "_labels", "_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv")
audio_folder = os.path.join(common_path, "_audio")
output_folder = os.path.join(output_path, "segmentLarge")

# Read the DataFrame
print(f"Reading CSV file: {csv_path}")
df1 = pd.read_csv(csv_path)
print(f"Loaded DataFrame with {len(df1)} rows.")
df = df1[(df1['language_tag'].isin(['Mandarin', 'English'])) & (df1['overlap_diff_lang'] == False)]
print(f"Filtered DataFrame with {len(df)} rows.")

# Initialize the AudioSegmenter
segmenter = AudioSegmenter(audio_folder, output_folder)
# Segment the audio files
segmented_df = segmenter.segment_audio_files(df)

#change here
output_pickle_path = os.path.join(output_path, "pickleLarge", "segmentLarge.pkl")
print(output_pickle_path)

print(f"Saving segmented DataFrame to: {output_pickle_path}")
with open(output_pickle_path, 'wb') as f:
    pickle.dump(segmented_df, f)
print("DataFrame saved successfully.")

print(segmented_df.shape)
# Count unique values in the language_tag column of the filtered DataFrame
print(segmented_df['language_tag'].value_counts())