import librosa
import numpy as np
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

# 파일 경로
mfcc_a_file = 'data/밝다/A.pkl'
mfcc_b_file = 'data/밝다/B.pkl'
voice_x_path = 'user_audio.wav'

mfcc_a_group = load_mfcc_from_file(mfcc_a_file)
mfcc_b_group = load_mfcc_from_file(mfcc_b_file)
mfcc_x = extract_mfcc(voice_x_path)
mfccStandard = mfcc_a_group[0]

sum_distance_to_a = sum_distance_to_group(mfcc_x, mfcc_a_group)
sum_distance_to_b = sum_distance_to_group(mfcc_x, mfcc_b_group)

score = similarity_scoring(mfcc_a_group, mfcc_x)

print("score: ", score)

print(f"Sum of DTW distances to A: {sum_distance_to_a}")
print(f"Sum of DTW distances to B: {sum_distance_to_b}")

# 단순 비교식
if sum_distance_to_a < sum_distance_to_b:
    print("Voice X is more similar to Voice Group A.")
else:
    print("Voice X is more similar to Voice Group B.")
