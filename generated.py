!pip install --q ipython-autotime
%load_ext autotime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

### 데이터셋 확장

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'This is a completely new document.',
    'A document is a piece of information.',
    'Is this your first document?',
    'Every document has its own purpose.',
    'The fourth document is quite different.',
    'Here is another unique document.',
    'Do you find this document useful?',
    'Documents can store valuable information.',
    'This is yet another example document.',
    'Some documents are very informative.',
    'Each document serves a specific purpose.',
]

labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 1, 0, 2]  # 문장 카테고리

# 2. Tokenizer로 텍스트 토큰화
vocab_size = 200
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)


X_train, X_val, y_train, y_val = train_test_split(padded, labels_categorical, test_size=0.2, random_state=42)

# 패딩
maxlen = 12
padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')


# 3. 라벨을 원-핫 인코딩
num_classes = len(set(labels))  # 클래스 개수
labels_categorical = to_categorical(labels, num_classes=num_classes)


# 4. 모델 정의
embedding_dim = 32  # 임베딩 벡터 크기


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 5. 모델 학습

# 데이터 분리

# 모델 학습
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val), verbose=1)

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# 모델 학습 시 클래스 가중치 적용
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val), class_weight=class_weights, verbose=1)

# model.fit(padded, labels_categorical, epochs=50, verbose=1, batch_size=8)


# 6. 새로운 문장 예측
new_sentences = [
    'This is a new document.',
    'Is this the second one?',
    'This document is completely new.',
    'Every document has its value.',
    'This is the fourth document and it is unique.',
    'Do you think this document is valuable?',
    'Some documents hold critical information.'
]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded = pad_sequences(new_sequences, maxlen=maxlen, padding='post', truncating='post')

predictions = model.predict(new_padded)

for i, sentence in enumerate(new_sentences):
    print(f"문장: '{sentence}' -> 예측된 카테고리: {np.argmax(predictions[i])}")