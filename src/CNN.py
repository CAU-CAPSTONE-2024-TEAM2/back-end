#mfcc 2D array(Time series), CNN model(ReLU, Softmax)
'''
CNN.
가장 정확도가 높을 것으로 예상되나
아직 모델의 파라미터 조정이나 활성화 함수의 결정이 곤란한 상태.
또한 mfcc의 차원 크기가 음성 파일에 따라 매우 달라
현재는 단순히 최고 차원 크기에 대해 padding하는 방법으로 전처리했지만
이걸 DTW와 합치는 방법을 구현하는 것이 좋겠음.
'''
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_accuracy',  # 모니터링 대상 지표
                               min_delta=0.0001,  # 이 값보다 작은 개선은 개선으로 간주하지 않음
                               patience=5,  # 지정된 에폭 수 동안 개선이 없으면 중단
                               verbose=1,  # 중단 시 로그 출력
                               mode='auto',  # 'auto', 'min', 'max' 중 선택. 'auto'는 자동으로 지표의 방향을 추정.
                               baseline=0.1,  # 이 값보다 성능이 좋아야 학습 계속 (val_accuracy 기준이므로)
                               restore_best_weights=True)  # 가장 좋은 모델의 가중치를 복원

# MFCC 추출 함수
def extract_mfcc(file_path, max_length=None):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if max_length:
        mfcc = librosa.util.fix_length(mfcc, size=max_length, axis=1)  # 최대 길이에 맞춰 패딩
    mfcc = np.expand_dims(mfcc, -1)  # CNN에 입력하기 위해 차원 추가
    return mfcc

# 데이터 준비 (여기서는 예시로써 'path_to_A_files', 'path_to_B_files'는 실제 경로로 대체해야 함)
# 'A'와 'B'의 음성 파일 경로 리스트
A_files = ['A1.mp3', 'A2.mp3']  # 예시 경로
B_files = ['B1.mp3', 'B2.mp3']

mfcc_lengths = []
for file in A_files + B_files:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_lengths.append(mfcc.shape[1])
    
max_length = max(mfcc_lengths)  # 최대 길이를 찾습니다.

# 데이터와 레이블 준비
X = []
y = []

for file in A_files:
    mfcc = extract_mfcc(file, max_length=max_length)
    X.append(mfcc)
    y.append(0)  # '발음 A'는 레이블 0으로 지정

for file in B_files:
    mfcc = extract_mfcc(file, max_length=max_length)
    X.append(mfcc)
    y.append(1)  # '발음 B'는 레이블 1으로 지정

X = np.array(X)
y = to_categorical(y)  # 레이블을 원-핫 인코딩

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN 모델 구성
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # '발음 A'와 '발음 B'를 분류하기 위한 출력 계층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 새로운 음성 '발음 X' 파일 분류
def predict_new_audio(file_path):
    new_mfcc = extract_mfcc(file_path, max_length=max_length)
    new_mfcc = np.expand_dims(new_mfcc, axis=0)  # 배치 차원 추가
    prediction = model.predict(new_mfcc)
    class_id = np.argmax(prediction)
    return '발음 A' if class_id == 0 else '발음 B'

# 예시 사용
print(predict_new_audio('X.mp3'))
