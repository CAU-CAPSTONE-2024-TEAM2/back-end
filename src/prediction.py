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

# 파일 경로 설정
mfcc_a_file = '../data_augmented/밝다/A.pkl'
mfcc_b_file = '../data_augmented/밝다/B.pkl'
voice_x_path = 'user_audio.wav'

# 데이터 로드
max_len = 100
mfcc_a_group = load_mfcc_from_file(mfcc_a_file, max_len)
mfcc_b_group = load_mfcc_from_file(mfcc_b_file, max_len)
mfcc_x = extract_mfcc(voice_x_path, max_len)

# 훈련 데이터 생성
mfcc_group = mfcc_a_group + mfcc_b_group
y_train = [0] * len(mfcc_a_group) + [1] * len(mfcc_b_group)

# 거리 행렬 계산
distance_matrix = compute_distance_matrix(mfcc_group)

# 새로운 샘플에 대한 거리 계산
distance_to_x = compute_distance_to_sample(mfcc_group, mfcc_x)
distance_to_a = compute_distance_to_sample(mfcc_a_group, mfcc_x)

# K-NN 분류기 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
knn.fit(distance_matrix, y_train)

# 예측
y_pred = knn.predict(distance_to_x.reshape(1, -1))
prob = knn.predict_proba(distance_to_x.reshape(1, -1))

# 결과 출력
if y_pred[0] == 0:
    print("Voice X is more similar to Voice Group A.")
else:
    print("Voice X is more similar to Voice Group B.")

print(f"Probability: {prob[0]}")

# 유사도 점수 계산
similarity_score = calculate_similarity_score(distance_to_a)

print(f"Score: {similarity_score}")

# 그래프 저장 함수
def save_mfcc_graph(mfcc, title, file_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


# 기준 MFCC 배열을 하나만 사용하여 그래프 저장
save_mfcc_graph(mfcc_a_group[0], 'Answer', 'mfcc_a.png')
save_mfcc_graph(mfcc_x, 'User', 'mfcc_x.png')
