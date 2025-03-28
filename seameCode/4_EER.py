import pandas as pd
from jiwer import wer, cer
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import ast  # Safe parsing for a list-like string

# Load the CSV data into DataFrames
df_conv = pd.read_csv('/export/c09/lavanya/languageIdentification/seame/segments/ConvExtracted.csv')
df_int = pd.read_csv('/export/c09/lavanya/languageIdentification/seame/segments/IntExtracted.csv')

# Function to calculate metrics for a given dataframe
def calculate_metrics(df):
    wer_list = []
    cer_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for index, row in df.iterrows():
        # Skip rows where 'extracted_transcript' is "Not found" or is NaN
        if row['extracted_transcript'] == "Not found" or pd.isna(row['extracted_transcript']):
            continue
        
        # Get the transcript text (reference) and extracted transcript (prediction)
        reference = row['transcript_text']
        
        try:
            # Use ast.literal_eval to safely parse the extracted_transcript into a list if possible
            prediction = ast.literal_eval(row['extracted_transcript']) if isinstance(row['extracted_transcript'], str) else row['extracted_transcript']
        except (ValueError, SyntaxError) as e:
            # If parsing fails, skip this row and log an error
            print(f"Skipping row {index} due to error parsing 'extracted_transcript': {e}")
            continue
        
        # If prediction is a float (e.g., NaN), skip this row
        if isinstance(prediction, float):
            continue
        
        # Ensure prediction is a list of words (if not, make it a list)
        if not isinstance(prediction, list):
            prediction = prediction.split()  # Assuming prediction is a space-separated string if it's not already a list
        
        # Calculate Word Error Rate (WER)
        wer_value = wer(reference, ' '.join(prediction))  # Convert prediction list to a string
        wer_list.append(wer_value)
        
        # Calculate Character Error Rate (CER)
        cer_value = cer(reference, ' '.join(prediction))  # Convert prediction list to a string
        cer_list.append(cer_value)
        
        # Tokenize the text to compare precision, recall, and F1 score
        ref_tokens = reference.split()  # Split the reference into words
        pred_tokens = prediction  # Already a list of words
        
        # Ensure the lengths match by padding/trimming
        max_len = max(len(ref_tokens), len(pred_tokens))
        ref_tokens = ref_tokens[:max_len] + [''] * (max_len - len(ref_tokens))  # Padding reference tokens if needed
        pred_tokens = pred_tokens[:max_len] + [''] * (max_len - len(pred_tokens))  # Padding prediction tokens if needed
        
        # Calculate Precision, Recall, and F1 score at the word level
        precision = precision_score(ref_tokens, pred_tokens, average='macro', zero_division=0)
        recall = recall_score(ref_tokens, pred_tokens, average='macro', zero_division=0)
        f1 = f1_score(ref_tokens, pred_tokens, average='macro', zero_division=0)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Calculate average WER and CER
    avg_wer = np.mean(wer_list)
    avg_cer = np.mean(cer_list)
    
    # Calculate average Precision, Recall, and F1 score
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    
    return avg_wer, avg_cer, avg_precision, avg_recall, avg_f1

# Calculate metrics for ConvExtracted (all rows)
avg_wer_conv, avg_cer_conv, avg_precision_conv, avg_recall_conv, avg_f1_conv = calculate_metrics(df_conv)

# Calculate metrics for IntExtracted (all rows)
avg_wer_int, avg_cer_int, avg_precision_int, avg_recall_int, avg_f1_int = calculate_metrics(df_int)

# Print the results for ConvExtracted
print("Metrics for ConvExtracted (all rows):")
print(f"Average WER: {avg_wer_conv}")
print(f"Average CER: {avg_cer_conv}")
print(f"Average Precision: {avg_precision_conv}")
print(f"Average Recall: {avg_recall_conv}")
print(f"Average F1 Score: {avg_f1_conv}")

# Print the results for IntExtracted
print("\nMetrics for IntExtracted (all rows):")
print(f"Average WER: {avg_wer_int}")
print(f"Average CER: {avg_cer_int}")
print(f"Average Precision: {avg_precision_int}")
print(f"Average Recall: {avg_recall_int}")
print(f"Average F1 Score: {avg_f1_int}")

# Optionally, save these metrics to a new CSV
metrics = {
    "Average WER (ConvExtracted)": avg_wer_conv,
    "Average CER (ConvExtracted)": avg_cer_conv,
    "Average Precision (ConvExtracted)": avg_precision_conv,
    "Average Recall (ConvExtracted)": avg_recall_conv,
    "Average F1 Score (ConvExtracted)": avg_f1_conv,
    "Average WER (IntExtracted)": avg_wer_int,
    "Average CER (IntExtracted)": avg_cer_int,
    "Average Precision (IntExtracted)": avg_precision_int,
    "Average Recall (IntExtracted)": avg_recall_int,
    "Average F1 Score (IntExtracted)": avg_f1_int,
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("metrics_output_all_rows.csv", index=False)
