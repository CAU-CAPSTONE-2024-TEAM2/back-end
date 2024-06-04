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


# MFCC 추출 함수
def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T


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


# 유사도 점수 계산 (0 ~ 100점)
def calculate_similarity_score(distances):
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    avg_distance = np.mean(distances)

    # 점수 계산: 거리가 작을수록 높은 점수
    score = 100 * (1 - (avg_distance - min_distance) / (max_distance - min_distance))
    return max(0, min(100, score))


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
