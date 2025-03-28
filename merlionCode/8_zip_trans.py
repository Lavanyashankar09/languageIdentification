import re
import csv

def parse_log_file(log_file_path, csv_file_path):
    """
    Parse the log file and extract audio file paths and their transcriptions.
    Write the results to a CSV file.
    
    Args:
        log_file_path (str): Path to the input log file
        csv_file_path (str): Path to output CSV file
    """
    # Dictionary to store file path -> transcription mappings
    results = {}
    current_file = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines containing file paths
            if line.strip().endswith('.wav:'):
                current_file = line.strip()[:-1]  # Remove the trailing colon
            
            # Look for transcription lines (they start with square brackets)
            elif current_file and line.strip().startswith('['):
                # Extract text between square brackets
                transcription = re.findall(r'\[(.*?)\]', line)
                if transcription:
                    # Join the words and store in results
                    results[current_file] = transcription[0]
                current_file = None  # Reset current file
    
    # Write results to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Audio File', 'Transcription'])
        # Write data
        for file_path, transcription in results.items():
            writer.writerow([file_path, transcription])
    
    print(f"Processing complete. Results written to {csv_file_path}")
    return results

# Example usage
if __name__ == "__main__":
    log_file_path = '/export/c09/lavanya/jobs/logs/merlion/zipformerExtract/log.zipformerLarge'
    csv_file_path = '/export/c09/lavanya/languageIdentification/merlion/zipformer_transcriptions.csv'
    
    results = parse_log_file(log_file_path, csv_file_path)