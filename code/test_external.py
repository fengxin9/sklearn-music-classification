#test_external.py 外部音频文件测试代码

"""
外部 wav 文件快速预测脚本
"""
import os
import librosa
import joblib
import numpy as np

MODEL_PATH = "gtzan_best_model.joblib"   # 主脚本训练完生成的模型包
TEST_WAV   = "test_song.wav"             # 放在项目根目录
N_MFCC     = 30                          # 与训练时保持一致

def extract_mfcc_mean(path, n_mfcc=N_MFCC, sr=22050, duration=30):
    #特征提取函数，与训练阶段相同
    y, sr = librosa.load(path, sr=sr, duration=duration)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def predict_audio(path):
    # 返回预测流派字符串, 各流派概率向量
    bundle = joblib.load(MODEL_PATH)
    model, genres = bundle["model"], bundle["genres"]
    feat = extract_mfcc_mean(path).reshape(1, -1)
    proba = model.predict_proba(feat)[0]          # 概率向量
    pred_id = np.argmax(proba)
    return genres[pred_id], proba

if __name__ == "__main__":
    if not os.path.isfile(TEST_WAV):
        exit(f"找不到测试文件：{TEST_WAV} ，请把它放到项目根目录再试")
    pred_genre, probas = predict_audio(TEST_WAV)
    print(f">>> 预测流派：{pred_genre}")
    print(">>> 各流派置信度（Top-5）:")
    for g, p in sorted(zip(joblib.load(MODEL_PATH)["genres"], probas),
                       key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {g:>10}: {p:.2%}")


'''
数据集外歌曲测试：
《Por una Cabeza》
>>> 预测流派：classical
>>> 各流派置信度（Top-5）:
   classical: 33.80%
     country: 19.60%
       blues: 13.40%
        jazz: 12.60%
      reggae: 8.80%
'''