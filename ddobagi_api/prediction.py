import os
import uuid

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


# 무성 구간 제거 함수
def remove_silence(y, sr, top_db=20):
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent_audio


# 에너지 계산 함수
def calculate_energy(y):
    return np.sum(y ** 2) / len(y)


# MFCC 추출 함수
def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path)
    y = remove_silence(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T, calculate_energy(y)


# 피클 파일에서 데이터 불러오기
def load_mfcc_from_file(file_path, max_len=100):
    with open(file_path, 'rb') as f:
        mfcc_group = pickle.load(f)
    padded_mfcc_group = []
    for mfcc in mfcc_group:
        if mfcc.shape[0] < max_len:
            mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]
        padded_mfcc_group.append(mfcc)
    return padded_mfcc_group



# DTW 거리 계산
def calculate_dtw(mfcc1, mfcc2):
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance


# 거리 행렬 계산
def compute_distance_matrix(mfcc_group):
    n = len(mfcc_group)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance = calculate_dtw(mfcc_group[i], mfcc_group[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


# 새로운 샘플에 대한 거리 계산
def compute_distance_to_sample(mfcc_group, mfcc_sample):
    distances = np.zeros(len(mfcc_group))
    for i in range(len(mfcc_group)):
        distances[i] = calculate_dtw(mfcc_group[i], mfcc_sample)
    return distances


# z-점수 기반 유사도 점수 계산 (0 ~ 100점)
def calculate_z_score_based_similarity(distances):
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    z_scores = (distances - mean_distance) / std_distance
    normalized_scores = 1 - (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))
    score = np.mean(normalized_scores) * 100

    return max(0, min(100, score))

# 에너지 기반 점수 조정
def adjust_score_for_energy(score, energy, threshold=0.004):
    print(f"energy: ", energy)
    if energy < threshold:
        return score * (energy / threshold)
    return score

# 그래프 저장 함수
def save_mfcc_graph(mfcc, title, file_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.close()
