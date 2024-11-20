import csv
import re

# Define the paths to the CSV and TXT files
csv_file_path = "/export/c09/lavanya/languageIdentification/seame/segments/ConvExtracted.csv"  # Path to the CSV file
txt_file_path = "/export/c09/lavanya/jobs/logs/seame/zipformer/zipformerConv.txt"  # Path to the TXT file containing logs

# Function to extract the relevant data from the log for a given segment
def extract_data_from_log(txt_file, segment_name):
    with open(txt_file, 'r', encoding='utf-8') as log_file:
        log_content = log_file.read()
        # Search for the segment name in the log content
        pattern = re.compile(rf"(/export/c09/lavanya/languageIdentification/seame/segments/conv/{segment_name}\.wav:)\s*(\[[^\]]*\])")
        match = pattern.search(log_content)
        if match:
            return match.group(2)  # Extract the matched data
    return None  # Return None if no match is found

# Open the CSV file and process each row
with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    rows = list(csv_reader)  # Read all rows into memory

    # Get header row and add a new column for the extracted transcript
    header = rows[0]
    header.insert(2, 'extracted_transcript')  # Insert new column for the extracted transcript in 3rd position
    
    # Iterate over each row in the CSV file (skip header)
    for i, row in enumerate(rows[1:], start=1):
        segment_filename = row[4].split('/')[-1]  # Extract the segment filename (e.g., NI01MAX_0101_1353_3612_Mandarin.wav)
        segment_name = segment_filename.replace('.wav', '')  # Remove .wav extension for matching
        
        # Extract data for this segment from the log file
        result = extract_data_from_log(txt_file_path, segment_name)
        
        # Insert the extracted transcript in the third column (position 2)
        if result:
            row.insert(2, result)  # Insert the extracted transcript in the 3rd column (index 2)
        else:
            row.insert(2, 'Not found')  # If no data is found, write 'Not found'
        
        # Optional: Print progress for large files
        if i % 100 == 0:
            print(f"Processed {i} rows")
    
    # Now write the updated rows back to the same CSV file
    with open(csv_file_path, 'w', encoding='utf-8', newline='') as writefile:
        csv_writer = csv.writer(writefile)
        csv_writer.writerows(rows)

print("Data extraction complete and saved in the same CSV.")
