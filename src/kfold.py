import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, KFold

# 무성 구간 제거 함수
def remove_silence(y, sr, top_db=20):
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent_audio

# MFCC 추출 함수
def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path)
    y = remove_silence(y, sr)
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

# 데이터 로드
max_len = 100
mfcc_a_group = load_mfcc_from_file('../data_augmented140/밝다/A.pkl', max_len)
mfcc_b_group = load_mfcc_from_file('../data_augmented140/밝다/B.pkl', max_len)

# 훈련 데이터 생성
mfcc_group = mfcc_a_group + mfcc_b_group
y_train = [0] * len(mfcc_a_group) + [1] * len(mfcc_b_group)

# 거리 행렬 계산
distance_matrix = compute_distance_matrix(mfcc_group)

# K-NN 분류기 생성 및 교차 검증
knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')

# 교차 검증 (K-Fold)
kf = KFold(n_splits=5)
cross_val_scores = cross_val_score(knn, distance_matrix, y_train, cv=kf, scoring='accuracy')

print("Cross-validation scores:", cross_val_scores)
print("Mean accuracy:", cross_val_scores.mean())
print("Standard deviation:", cross_val_scores.std())

# 새로운 샘플에 대한 거리 계산
mfcc_x = extract_mfcc('../발따.wav', max_len)
distance_to_x = np.array([calculate_dtw(mfcc, mfcc_x) for mfcc in mfcc_group]).reshape(1, -1)

# K-NN 학습 및 예측
knn.fit(distance_matrix, y_train)
y_pred = knn.predict(distance_to_x)
prob = knn.predict_proba(distance_to_x)

# 결과 출력
if y_pred[0] == 0:
    print("Voice X is more similar to Voice Group A.")
else:
    print("Voice X is more similar to Voice Group B.")

print(f"Probability: {prob[0]}")

# 혼동 행렬 및 분류 보고서
y_train_pred = knn.predict(distance_matrix)
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Classification Report:")
print(classification_report(y_train, y_train_pred))
