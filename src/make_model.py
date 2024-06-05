import librosa
import numpy as np
import pickle
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump


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


# 데이터 디렉터리
data_dir = '../data_augmented'
model_dir = '../model'

# data_augmented 폴더 내의 모든 하위 폴더를 반복
for word_folder in os.listdir(data_dir):
    word_folder_path = os.path.join(data_dir, word_folder)

    if os.path.isdir(word_folder_path):
        mfcc_a_file = os.path.join(word_folder_path, 'A.pkl')
        mfcc_b_file = os.path.join(word_folder_path, 'B.pkl')

        if os.path.exists(mfcc_a_file) and os.path.exists(mfcc_b_file):
            # MFCC 데이터 로드
            max_len = 100
            mfcc_a_group = load_mfcc_from_file(mfcc_a_file, max_len)
            mfcc_b_group = load_mfcc_from_file(mfcc_b_file, max_len)

            # 훈련 데이터 생성
            mfcc_group = mfcc_a_group + mfcc_b_group
            y_train = [0] * len(mfcc_a_group) + [1] * len(mfcc_b_group)

            # 거리 행렬 계산
            distance_matrix = compute_distance_matrix(mfcc_group)

            # K-NN 분류기 생성 및 학습
            knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
            knn.fit(distance_matrix, y_train)

            # 모델 저장 경로
            model_word_dir = os.path.join(model_dir, word_folder)
            os.makedirs(model_word_dir, exist_ok=True)

            # 모델 저장
            dump(knn, os.path.join(model_word_dir, 'knn_model.joblib'))
            dump(distance_matrix, os.path.join(model_word_dir, 'distance_matrix.joblib'))
            dump(y_train, os.path.join(model_word_dir, 'y_train.joblib'))

            print(f"K-NN model and distance matrix for '{word_folder}' saved to '{model_word_dir}'")
