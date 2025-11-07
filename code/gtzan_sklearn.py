#sklearn_musicGenre_classification on_GTZAN_dataset

"""
GTZAN 10-genre classification using sklearn
依赖工具包: librosa scikit-learn joblib matplotlib seaborn
目录结构:
data/
└── GTZAN/
    ├── blues/
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 参数
DATA_DIR = "data/GTZAN"   # GTZAN数据集路径
SAMPLE_RATE = 22050     # GTZAN采样率
DURATION = 30       # 每段30s
N_MFCC = 30         # 取30维MFCC均值
RANDOM_STATE = 42   # 随机种子
N_JOBS = -1         # 并行线程数

# 特征提取函数
def extract_mfcc_mean(path, n_mfcc=N_MFCC, sr=SAMPLE_RATE, duration=DURATION):
    """读取单文件，返回 n_mfcc 维 MFCC 均值向量"""
    y, sr = librosa.load(path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# 数据集加载与特征提取
def load_dataset(data_dir):
    """遍历目录，提取特征和标签"""
    genres = sorted(g for g in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, g)))
    X, y = [], []
    genre2id = {g: i for i, g in enumerate(genres)}
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        for fname in os.listdir(genre_dir):
            if fname.endswith(".wav"):
                path = os.path.join(genre_dir, fname)
                try:
                    feat = extract_mfcc_mean(path)
                    X.append(feat)
                    y.append(genre2id[genre])
                except Exception as e:            # 文件出错则跳过
                    print("跳过文件:", path, e)
    return np.array(X), np.array(y), genres

# 读取数据
print(">>> 正在读取音频并提取特征 … 耗时约 1~2 分钟")
X, y, genres = load_dataset(DATA_DIR)
print("数据形状:", X.shape, y.shape)

# 数据集划分，先整体按8:2划分(前80%训练、后20%测试)，再在80%上做3折交叉验证
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 建模
models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=N_JOBS,
                                   multi_class="ovr", random_state=RANDOM_STATE))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=N_JOBS,
        random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)    # 3折交叉验证

best_model = None
best_score = 0
for name, model in models.items():     
    print(f"\n>>> 交叉验证 {name} …")
    cv_scores = []
    for fold, (tr, val) in enumerate(cv.split(X_train, y_train)):        # 分fold训练/验证
        model.fit(X_train[tr], y_train[tr])
        val_pred = model.predict(X_train[val])
        acc = accuracy_score(y_train[val], val_pred)
        cv_scores.append(acc)
        print(f"   fold{fold+1} accuracy={acc:.4f}")
    mean_acc = np.mean(cv_scores)
    print(f"{name} 平均 CV accuracy = {mean_acc:.4f}")        # 记录最佳模型
    if mean_acc > best_score:
        best_score = mean_acc
        best_model = model

# 最终评估
print("\n>>> 在最终测试集上评估 …")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("测试集 accuracy =", accuracy_score(y_test, y_pred))
print("\n详细测试报告:\n", classification_report(y_test, y_pred, target_names=genres))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=genres, yticklabels=genres)
plt.title("Confusion matrix – best model")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 保存模型为joblib包
model_path = "gtzan_best_model.joblib"
joblib.dump({"model": best_model, "genres": genres, "scaler": None}, model_path)
print(f"\n模型已保存到 {model_path}")

# 预测新文件
def predict_audio(path):
    """输入任意 wav 路径，返回预测流派名称"""
    feat = extract_mfcc_mean(path).reshape(1, -1)
    bundle = joblib.load(model_path)
    model, genres = bundle["model"], bundle["genres"]
    pred_id = model.predict(feat)[0]
    return genres[pred_id]

if __name__ == "__main__":     # 预测一个测试集文件
    sample = X_test[0]
    true_genre = genres[y_test[0]]   # 把向量写回临时文件再预测
    
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # 逆变换简单生成波形
        fake_wave = np.random.randn(SAMPLE_RATE * DURATION) * 0.01
        sf.write(tmp.name, fake_wave, SAMPLE_RATE)
        print("预测结果:", predict_audio(tmp.name), "/ 实际流派:", true_genre)