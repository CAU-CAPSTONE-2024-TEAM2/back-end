'''
CNN.
CNN.py에 dtw의 가중치를 더한 예제
정확도가 낮음
'''
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from dtw import accelerated_dtw


# 데이터 증강 함수
def augment_data(y, sr):
    # 시간 이동
    shift = np.random.randint(sr)
    augmented_data = np.roll(y, shift)

    # 피치 변조
    pitch_factor = np.random.uniform(-1, 1)
    augmented_data = librosa.effects.pitch_shift(augmented_data, sr=sr, n_steps=pitch_factor)

    # 속도 변조
    speed_factor = np.random.uniform(0.9, 1.1)
    augmented_data = librosa.effects.time_stretch(augmented_data, rate=speed_factor)

    # 노이즈 추가
    noise = np.random.randn(len(augmented_data))
    augmented_data = augmented_data + 0.005 * noise

    return augmented_data


# 데이터 로드 및 전처리 함수
def load_and_augment_data(data_path, sr=22050, n_mfcc=13, max_len=100, augmentation_factor=5):
    labels = []
    features = []

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                y, _ = librosa.load(file_path, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                # MFCC 길이를 최대 길이(max_len)에 맞추기
                padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
                features.append(padded_mfcc)
                labels.append(label)

                # 데이터 증강
                for _ in range(augmentation_factor):
                    augmented_y = augment_data(y, sr)
                    augmented_mfcc = librosa.feature.mfcc(y=augmented_y, sr=sr, n_mfcc=n_mfcc)
                    padded_augmented_mfcc = np.pad(augmented_mfcc, ((0, 0), (0, max_len - augmented_mfcc.shape[1])),
                                                   mode='constant')
                    features.append(padded_augmented_mfcc)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# 기준 MFCC 설정
def get_reference_mfcc(features, labels, target_label):
    target_indices = [i for i, label in enumerate(labels) if label == target_label]
    return features[target_indices[0]]  # 첫 번째 해당 라벨을 기준으로 사용 (필요에 따라 변경 가능)
# 0: 박따 / 1: 발따

# DTW를 사용하여 MFCC 정렬
def align_mfccs(mfcc, ref_mfcc, max_len=100):
    aligned_mfcc = []
    for feature in mfcc:
        dist, cost, acc_cost, path = accelerated_dtw(feature.T, ref_mfcc.T, dist='euclidean')
        aligned_feature = np.zeros((ref_mfcc.shape[0], max_len))
        for i, (ix, iy) in enumerate(zip(path[0], path[1])):
            if i < max_len:
                aligned_feature[:, i] = feature[:, ix]
        aligned_mfcc.append(aligned_feature)
    return np.array(aligned_mfcc)


# 데이터 경로
data_path = "data"

# 데이터 로드 및 증강
features, labels = load_and_augment_data(data_path)

# 기준 MFCC 설정 (여기서는 "baktta"를 기준으로 사용)
ref_mfcc = get_reference_mfcc(features, labels, target_label="baktta")

# 모든 MFCC 정렬
max_len = 100
aligned_features = align_mfccs(features, ref_mfcc, max_len)

# 라벨 인코딩
label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
encoded_labels = np.array([label_dict[label] for label in labels])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(aligned_features, encoded_labels, test_size=0.2, random_state=42)

# 데이터 형상 변경 (CNN 입력을 위해 4D로 변경)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 라벨 원-핫 인코딩
y_train = to_categorical(y_train, num_classes=len(label_dict))
y_test = to_categorical(y_test, num_classes=len(label_dict))

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(13, max_len, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# 예측 결과 출력
predicted_probabilities = model.predict(X_test)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# 예측 결과와 실제 라벨 비교
for i, (predicted_label, true_label) in enumerate(zip(predicted_labels, y_test)):
    print(f"Sample {i+1}: Predicted label - {predicted_label}, True label - {true_label}")

# 테스트 세트 정확도 계산
accuracy = np.sum(predicted_labels == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# 사용자 음성 파일 경로
user_audio_path = "user_audio.wav"

# 사용자 음성 파일 로드 및 MFCC 추출
sr = 22050  # 샘플링 레이트 설정
n_mfcc = 13  # 추출할 MFCC 특징의 수 설정

# 사용자 음성 파일 불러오기
user_audio, _ = librosa.load(user_audio_path, sr=sr)

# MFCC 추출
user_mfcc = librosa.feature.mfcc(y=user_audio, sr=sr, n_mfcc=n_mfcc)

# MFCC 길이 맞추기
max_len = 100  # 모델 입력에 맞추기 위한 최대 길이
padded_user_mfcc = np.pad(user_mfcc, ((0, 0), (0, max_len - user_mfcc.shape[1])), mode='constant')
padded_user_mfcc = padded_user_mfcc[np.newaxis, ..., np.newaxis]  # 모델 입력 형태로 변환

# 모델을 사용하여 레이블 예측
predicted_label = np.argmax(model.predict(padded_user_mfcc))

# 테스트 세트 정확도 계산
accuracy = np.sum(predicted_labels == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# 예측된 레이블 출력
print("Predicted label:", predicted_label)
