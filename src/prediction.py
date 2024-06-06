import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
import time

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

start = time.time()

# 모델 및 거리 행렬 로드
knn = load('../model/밝다/knn_model.joblib')
distance_matrix = load('../model/밝다/distance_matrix.joblib')
y_train = load('../model/밝다/y_train.joblib')

# 파일 경로 설정
mfcc_a_file = '../data_augmented/밝다/A.pkl'
mfcc_b_file = '../data_augmented/밝다/B.pkl'
voice_x_path = ('../박따.wav')

# 데이터 로드
max_len = 100
mfcc_a_group = load_mfcc_from_file('../data_augmented/밝다/A.pkl', max_len)
mfcc_b_group = load_mfcc_from_file('../data_augmented/밝다/B.pkl', max_len)
mfcc_x, energy_x = extract_mfcc(voice_x_path, max_len)

# 새로운 샘플에 대한 거리 계산
distance_to_x = compute_distance_to_sample(mfcc_a_group + mfcc_b_group, mfcc_x)
distance_to_a = compute_distance_to_sample(mfcc_a_group, mfcc_x)

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

z_similarity_score = calculate_z_score_based_similarity(distance_to_a)

# 에너지 기반 점수 조정
final_score3 = adjust_score_for_energy(z_similarity_score, energy_x)

print(f"Score3: {z_similarity_score}")
print(f"Score3-1: {final_score3}")

end = time.time()

print(f"{end - start:.5f} sec")
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
