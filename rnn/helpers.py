from os import walk
import numpy as np
import matplotlib.pyplot as plt

#imports for WAV2VEC2
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd
import librosa

import os
# parse the file names into a numpy matrix, each row represents one file.
RAW_DATA_PATH = '../data/raw'

def get_file_matrix():
    dirs = []
    for (dirpath, dirnames, _) in walk(RAW_DATA_PATH):
        for d in dirnames:
            if d.startswith("Actor"):
                dirs.append(dirpath+'/'+d)
        break
        
    all_files = []
    for actor in dirs:
        for(_,_,files) in walk(actor):
            all_files+=[i.split('.')[0].split('-') for i in files]
            break
    all_files = [[int(i) - 1 for i in row] for row in all_files]

    files = np.array(all_files)
    print(files)
    return files


def matrix_to_filename(matrix, RAW_DATA_PATH):
    filepaths = []
    print(matrix)
    for row in matrix:
        filename = ''
        for col in row:
            filename+=str(col+1).zfill(2)
            filename+='-'
        filename = filename[:-1]
        filename+='.wav'
        filepaths.append(RAW_DATA_PATH+'/Actor_'+str(row[6]+1).zfill(2)+'/'+filename)
        # print(filename)
    return filepaths

def get_labels(matrix):
    labels = []
    for row in matrix:
        labels.append(row[2])
    return labels
def plot_counts(files):
    print(f"Total number of files {files.shape[0]}")

    labels = [['full-AV', 'video-only', 'audio-only'],
             ['speech', 'song'],
             ['neutral' , 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust','surprised'],
             ['normal', 'strong'],
             ['Kids are talking by the door', 'Dogs are sitting by the door'],
             ['1st repetition', '2nd repetition'],
             ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
             ]
    titles = ['Modality', 'Vocal channel', 'Emotion', 'Emotional intensity' ,'Statement','Repetition' ,'Actor']
    
    fig, ax = plt.subplots(2, 4, figsize=(20, 10)) 
    inds = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]
    for i in range(7):
        unique, counts = np.unique(files[:, i], return_counts = True)
        unique = np.array(labels[i])[unique]
        ax[inds[i]].bar(unique, counts)
        ax[inds[i]].set_title(titles[i])
    ax[inds[-1]].axis('off')
    plt.suptitle("Count Plots")
    plt.tight_layout()
    plt.show()

#device
device = "cuda" if torch.cuda.is_available() else "cpu"

#Load the pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", dtype=torch.float16, attn_implementation="flash_attention_2")

#model
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)

#function for getting embeddings
def get_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    
    #Wav2Vec2 expects 16kHz sample rate so resample 
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    #convert to model input format
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    
    #Extract the embeddings 
    with torch.no_grad():
        outputs = model(**inputs)
    
    #Take mean across time dimension to get single vector representation
    embedding = outputs.last_hidden_state
    return embedding.cpu().numpy()

def graph(train_acc, test_acc, train_loss, test_loss, epochs):
    plt.figure(figsize=(10, 6)) 
    plt.plot(list(range(epochs)), train_acc, label='Training Accuracy')
    plt.plot(list(range(epochs)), test_acc, label='Test Accuracy')
    plt.legend()
    plt.savefig("accuracy.png") 
    plt.clf()
    plt.figure(figsize=(10, 6)) 
    plt.plot(list(range(epochs)), train_loss, label='Training loss')
    plt.plot(list(range(epochs)), test_loss, label='Test loss')
    plt.legend()
    plt.savefig("loss.png") 
    plt.clf()
def get_wav2vec_embeddings(EMBEDDING_FILE, file_paths):
    embeddings = []
    print("Getting embeddings")

    if(os.path.exists(EMBEDDING_FILE)):
        print(f"Loading embeddings from {EMBEDDING_FILE}")
        embeddings = torch.load(EMBEDDING_FILE)
    else:
        print(f"Loading embeddings using Wav2Vec")
        c = 0
        for f in file_paths:
            embeddings.append(torch.from_numpy(get_embedding(f)).unsqueeze(-1).squeeze(0).squeeze(-1))
            c+=1
            print(c)
        torch.save(embeddings, EMBEDDING_FILE)
        print(f"Saved embeddings to {EMBEDDING_FILE}")
    print("Done getting embeddings")
    return embeddings