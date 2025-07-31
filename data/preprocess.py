import os
import re
import csv


# 1. Generate metadata
# Define paths for the sessions
input_folder_pattern = "./IEMOCAP_full_release/Session{}/dialog/EmoEvaluation"
transcription_folder_pattern = "./IEMOCAP_full_release/Session{}/dialog/transcriptions"
audio_folder_pattern = "./IEMOCAP_full_release/Session{}/sentences/wav"


#  Extract emotion data from the evaluation files
def extract_emotion_data(session):
    input_folder = input_folder_pattern.format(session)
    emotion_data = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = re.match(r"\[(.*?)\]\s+(\S+)\s+(\S+)", line)
                        if match:
                            time = match.group(1)
                            fileid = match.group(2)
                            emotion = match.group(3)
                            emotion_data.append({
                                'fileid': fileid,
                                'time': time,
                                'emotion': emotion
                            })
    return emotion_data


# Extract transcription data and handle `prev` and `next`
def extract_transcription_data(session):
    transcription_folder = transcription_folder_pattern.format(session)
    transcription_data = {}
    for root, _, files in os.walk(transcription_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                prev_fileid = None
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        # Match valid fileid lines (e.g., "Ses01F_impro01_F000")
                        match = re.match(r"^(\S+)\s+\[.*?\]:\s+(.*)", line)
                        if match:
                            fileid = match.group(1)
                            transcription = match.group(2)

                            # Find the next valid fileid line
                            next_fileid = None
                            for j in range(i + 1, len(lines)):
                                next_match = re.match(r"^(\S+)\s+\[.*?\]:", lines[j])
                                if next_match:
                                    next_fileid = next_match.group(1)
                                    break

                            transcription_data[fileid] = {
                                'transcription': transcription,
                                'prev': prev_fileid,
                                'next': next_fileid
                            }
                            prev_fileid = fileid
    return transcription_data


# Step 3: Combine emotion data with transcription data
def combine_emotion_and_transcription(emotion_data, transcription_data):
    combined_data = []
    for row in emotion_data:
        fileid = row['fileid']
        transcription_info = transcription_data.get(fileid, {'transcription': None, 'prev': None, 'next': None})
        combined_data.append({
            'fileid': fileid,
            'time': row['time'],
            'emotion': row['emotion'],
            'transcription': transcription_info['transcription'],
            'prev': transcription_info['prev'],
            'next': transcription_info['next']
        })
    return combined_data


#Add audiopath to the combined data
def add_audiopath_to_data(combined_data, session):
    audio_folder = audio_folder_pattern.format(session)
    for row in combined_data:
        fileid = row['fileid']
        base_dir = "_".join(fileid.split("_")[:-1])
        audiopath = os.path.join(audio_folder, base_dir, f"{fileid}.wav")
        row['audiopath'] = audiopath if os.path.exists(audiopath) else None
    return combined_data


# Process all sessions and save to CSV
def process_sessions_to_csv(output_csv):
    all_data = []
    for session in range(1, 6):  # Loop through Session1 to Session5
        print(f"Processing Session {session}...")
        # Extract emotion data
        emotion_data = extract_emotion_data(session)
        # Extract transcription data and handle prev/next
        transcription_data = extract_transcription_data(session)
        # Combine emotion and transcription data
        combined_data = combine_emotion_and_transcription(emotion_data, transcription_data)
        # Add audiopath information
        final_data = add_audiopath_to_data(combined_data, session)
        all_data.extend(final_data)

    # Write all combined data to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file,
                                fieldnames=['fileid', 'audiopath', 'prev', 'next', 'time', 'emotion', 'transcription'])
        writer.writeheader()
        writer.writerows(all_data)


# Output CSV path
output_csv = "IEMOCAP_extracted_emotions.csv"

# Process and save data
process_sessions_to_csv(output_csv)

print(f"Data processed and saved to {output_csv}")

# 2. Use CLAP to generate audio and text features

import sys
sys.argv = ['']

import numpy as np
import librosa
import torch
import laion_clap

# load CLAP model
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

print(model)

# generate Audio features
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


#  Read the CSV file
csv_file_path = "IEMOCAP_extracted_emotions.csv"
df = pd.read_csv(csv_file_path)
file_ids = df["fileid"].tolist()  #  fileid
# Dynamically generate the emotion label mapping table
emotion_labels = df['emotion'].unique()
emotion_label_map = {emotion: idx for idx, emotion in enumerate(sorted(emotion_labels))}
print("Emotion label mapping table:", emotion_label_map)

# Extract features and labels
feature_list = []
label_list = []

print("Extracting audio features...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    audiopath = row['audiopath']
    emotion = row['emotion']
#     print(audiopath)
    # Skip invalid paths or emotions
    if not os.path.exists(audiopath) or emotion not in emotion_label_map:
        print("No file")
        continue

    # Extract audio features using openSMILE
    features = model.get_audio_embedding_from_filelist(x = [audiopath])

    # Add features and labels
    feature_list.append(features)
    label_list.append(emotion_label_map[emotion])
#     break
# Convert to arrays
features_array = np.array(feature_list)  # Shape: [num_samples, feature_dim]
labels_array = np.array(label_list)      # Shape: [num_samples]
final_file_ids = np.array(file_ids)


# Save features and labels
output_path = "IEMOCAP_audio_features_512.npy"
np.save(output_path, {"fileid": final_file_ids, "features": features_array, "labels": labels_array})
print(f"Feature extraction completed and saved to {output_path}")

# Generate Text feature
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm



# Process data, extract context, and labels
def process_csv_and_generate_embeddings(csv_path, output_path):
    """
    Extract context embeddings from a CSV file and save to a .npy file
    """
    df = pd.read_csv(csv_path)

    # Dynamically generate emotion label mappings
    emotion_labels = df['emotion'].unique()
    emotion_label_map = {emotion: idx for idx, emotion in enumerate(sorted(emotion_labels))}
    print("Emotion label mapping table:", emotion_label_map)

    feature_list = []
    label_list = []

    print("Extracting context-aware embeddings...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        target_sentence = row['transcription']
        #prev_id = row['prev']
        #next_id = row['next']
        # Extract context-aware embedding
        embedding = model.get_text_embedding(target_sentence)
        feature_list.append(embedding)
        label_list.append(emotion_label_map[row['emotion']])

    # Convert to arrays
    features_array = np.array(feature_list)  # Shape: [num_samples, 768]
    labels_array = np.array(label_list)      # Shape: [num_samples]

    # Save as a dictionary
    final_data = {"features": features_array, "labels": labels_array}
    np.save(output_path, final_data)
    print(f"Feature extraction and saving completed. File saved at: {output_path}")

    return target_sentence

# File paths
csv_path = "IEMOCAP_extracted_emotions.csv"
output_path = "IEMOCAP_text_features_512.npy"

# Extract embeddings and save
process_csv_and_generate_embeddings(csv_path, output_path)

# 4. Generate the conversation block file
# audio

import pandas as pd

audio_data = np.load("./IEMOCAP_audio_features_512.npy", allow_pickle=True).item()
audio_features = audio_data['features']
audio_labels = audio_data['labels']
zeros_array = np.zeros(768)
previous_com_features = []
metadata = pd.read_csv('./IEMOCAP_extracted_emotions.csv')

file_to_prev = dict(zip(metadata['fileid'], metadata['prev']))

for file in audio_data['fileid']:
    current_gender = file[-4]
    prev = file_to_prev[file]
    feature = zeros_array
    while not pd.isna(prev):
        if prev[-4] == current_gender:
            try:
                prev = file_to_prev[prev]
            except:
                break
        else:
            try:
                position = np.where(audio_data['fileid'] == prev)[0][0]
                feature = audio_data['features'][position]
                break
            except:
                break

    previous_com_features.append(feature)

previous_com_features = np.vstack(previous_com_features)
audio_data['oppo_prev'] = previous_com_features
np.save("./clap_with_oppo_IEMOCAP_text_features_512.npy", audio_data, allow_pickle=True)


# text
text_data = np.load("./IEMOCAP_audio_features_512.npy", allow_pickle=True).item()
text_features = text_data['features']
text_labels = text_data['labels']

import pandas as pd

zeros_array = np.zeros(768)
previous_com_features = []
metadata = pd.read_csv('./IEMOCAP_extracted_emotions.csv')

file_to_prev = dict(zip(metadata['fileid'], metadata['prev']))

for file in text_data['fileid']:
    current_gender = file[-4]
    prev = file_to_prev[file]
    feature = zeros_array
    while not pd.isna(prev):
        if prev[-4] == current_gender:
            try:
                prev = file_to_prev[prev]
            except:
                break
        else:
            try:
                position = np.where(text_data['fileid'] == prev)[0][0]
                feature = text_data['features'][position]
                break
            except:
                break

    previous_com_features.append(feature)
previous_com_features = np.vstack(previous_com_features)
text_data['oppo_prev'] = previous_com_features
np.save("./clap_with_oppo_IEMOCAP_text_features_512.npy", text_data, allow_pickle=True)

