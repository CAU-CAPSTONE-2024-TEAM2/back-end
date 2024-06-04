import librosa
import numpy as np
import pickle
import os
import random


# 피클 파일에서 데이터 불러오기
def load_mfcc_from_file(file_path):
    with open(file_path, 'rb') as f:
        mfcc_group = pickle.load(f)
    return mfcc_group


# 피클 파일에 데이터 저장하기
def save_mfcc_to_file(mfcc_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(mfcc_list, f)


# MFCC 데이터 증강
def augment_mfcc(mfcc):
    augmented_mfccs = []

    # 시간축 이동
    shift_range = random.randint(-2, 2)
    augmented_mfccs.append(np.roll(mfcc, shift_range, axis=0))

    # 주파수 축 이동
    shift_range = random.randint(-2, 2)
    augmented_mfccs.append(np.roll(mfcc, shift_range, axis=1))

    # 가우시안 노이즈 추가
    noise = np.random.normal(0, 0.1, mfcc.shape)
    augmented_mfccs.append(mfcc + noise)

    # 크기 변환 (작게 혹은 크게)
    scale_factor = random.uniform(0.9, 1.1)
    augmented_mfccs.append(mfcc * scale_factor)

    return augmented_mfccs


# MFCC 데이터 증강 적용
def augment_mfcc_group(mfcc_group, target_size=10):
    augmented_group = []
    for mfcc in mfcc_group:
        augmented_group.append(mfcc)  # 원본 추가
        while len(augmented_group) < target_size:
            augmented_group.extend(augment_mfcc(mfcc))
            if len(augmented_group) >= target_size:
                break
    return augmented_group[:target_size]


# 원본 데이터 폴더와 증강 데이터 폴더
data_dir = '../data'
augmented_data_dir = '../data_augmented'

# data 폴더 내의 모든 하위 폴더를 반복
for word_folder in os.listdir(data_dir):
    word_folder_path = os.path.join(data_dir, word_folder)

    if os.path.isdir(word_folder_path):
        mfcc_a_file = os.path.join(word_folder_path, 'A.pkl')
        mfcc_b_file = os.path.join(word_folder_path, 'B.pkl')

        if os.path.exists(mfcc_a_file) and os.path.exists(mfcc_b_file):
            # MFCC 데이터 로드
            mfcc_a_group = load_mfcc_from_file(mfcc_a_file)
            mfcc_b_group = load_mfcc_from_file(mfcc_b_file)

            # 데이터 증강
            augmented_mfcc_a_group = augment_mfcc_group(mfcc_a_group, target_size=10)
            augmented_mfcc_b_group = augment_mfcc_group(mfcc_b_group, target_size=10)

            # 증강된 MFCC 데이터 저장
            save_mfcc_to_file(augmented_mfcc_a_group, os.path.join(augmented_data_dir, word_folder, 'A.pkl'))
            save_mfcc_to_file(augmented_mfcc_b_group, os.path.join(augmented_data_dir, word_folder, 'B.pkl'))

            print(
                f"Augmented MFCC features for '{word_folder}' saved to '{augmented_data_dir}/{word_folder}/A.pkl' and '{augmented_data_dir}/{word_folder}/B.pkl'")
