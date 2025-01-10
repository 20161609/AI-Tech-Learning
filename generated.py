from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 데이터셋 생성: 노이즈와 복잡성 증가
X, y = make_classification(
    n_samples=1000,
    n_features=30,          # 총 30개의 특성
    n_informative=10,       # 유용한 특성 10개
    n_redundant=5,          # 상관된 특성 5개
    n_clusters_per_class=2, # 클래스당 클러스터 개수
    flip_y=0.15,            # 15% 레이블 플립
    random_state=42
)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 기본 SVM 모델
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 데이터 표준화
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))  # 기본 설정
])

# 기본 모델 학습 및 평가
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("기본 모델 성능:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 하이퍼파라미터 튜닝
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100, 1000],     # 더 넓은 C 값
    'svm__gamma': [0.001, 0.01, 0.1, 1, 10],     # 다양한 gamma 값
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # 다양한 커널
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 튜닝된 모델 평가
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\n튜닝된 모델 성능:")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Classification Report:\n", classification_report(y_test, y_pred_tuned))
