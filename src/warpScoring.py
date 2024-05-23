import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# mfcc 추출
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc.T  # Transpose for DTW

# fast DTW 계산식
def calculate_dtw(mfcc1, mfcc2):
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

# dtw 합산
def sum_distance_to_group(mfcc_x, group_mfccs):
    distances = [calculate_dtw(mfcc_x, mfcc) for mfcc in group_mfccs]
    return np.sum(distances)

# 피클 파일에서 데이터 불러오기
def load_mfcc_from_file(file_path):
    with open(file_path, 'rb') as f:
        mfcc_group = pickle.load(f)
    return mfcc_group

# 유사도 계산식
def similarity_scoring(standard, user):
    sum = 0
    l = len(standard)
    for mfcc in standard:
        cosine_sim = cosine_similarity(user, mfcc)
        sum += np.mean(cosine_sim) * 100
    return sum / l

# MFCC 그래프 저장
def save_mfcc_graph(mfcc, title, file_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

# 파일 경로
mfcc_a_file = 'data/밝다/A.pkl'
mfcc_b_file = 'data/밝다/B.pkl'
voice_x_path = 'user_audio.wav'

mfcc_a_group = load_mfcc_from_file(mfcc_a_file)
mfcc_b_group = load_mfcc_from_file(mfcc_b_file)
mfcc_x = extract_mfcc(voice_x_path)

sum_distance_to_a = sum_distance_to_group(mfcc_x, mfcc_a_group)
sum_distance_to_b = sum_distance_to_group(mfcc_x, mfcc_b_group)

# 기준 MFCC 배열을 하나만 사용하여 그래프 저장
# mfcc_a.png가 정답 이미지, mfcc_x.png가 유저 이미지
save_mfcc_graph(mfcc_a_group[0].T, 'Answer', 'mfcc_a.png')
save_mfcc_graph(mfcc_x.T, 'User', 'mfcc_x.png')

score = similarity_scoring(mfcc_a_group, mfcc_x)

# accuracy 점수
print("score: ", score)

# test 출력 코드
print(f"Sum of DTW distances to A: {sum_distance_to_a}")
print(f"Sum of DTW distances to B: {sum_distance_to_b}")

# 단순 비교식(유사 1-NN)
# sum_distance_to 값이 큰 쪽으로 분류
if sum_distance_to_a < sum_distance_to_b:
    print("Voice X is more similar to Voice Group A.")
else:
    print("Voice X is more similar to Voice Group B.")