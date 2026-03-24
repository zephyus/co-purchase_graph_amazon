# 第七章：考試與複習的藝術（訓練集、驗證集與測試集概念）

## 1. The Intuition (引言與核心靈魂)

想像你正在準備高中的期末考，你拿到了一本含有 1000 題的「歷屆考古題題庫」。
如果你把這 1000 題從頭到尾死背下來（包含答案是 A 還是 B），你可以在寫這本題庫時拿到 100 分。但真正的期末考到來時，老師出了「沒見過的新題目」，如果你只會死背，你肯定會考 0 分。
這在機器學習中被稱為**過擬合 (Overfitting)**：模型「死背」了歷史資料，卻失去了「舉一反三 (Generalization)」的能力。

為了防止這個災難，聰明的學生會這樣讀書：
1. **平時練習 (Training Set / 訓練集)**：從題庫拿出 700 題，看著答案練習，試著找出解題規律。
2. **模擬考 (Validation Set / 驗證集)**：拿出 200 題當作模擬考，不先看答案。根據模擬考的分數，來微調自己的讀書方法（這叫做 tuning hyperparameters，調超參數）。
3. **終極期末考 (Test Set / 測試集)**：剩下的 100 題，**絕對不能偷看**，直到學期最後一天才拿出來測驗。這 100 題的成績，就是你真實實力的客觀證明。

你的 Amazon GNN 作業，就嚴格切分了 `train_edges`, `val_edges` 以及 `test_edges`。如果不這麼做，所有的分數都會是一場騙局。

### Learning Objectives (學習目標)
1. **防禦過擬合**：明白大腦死背資料的風險，並學會畫出 Train Loss 與 Val Loss 的學習曲線。
2. **理解資料三重切分**：徹底弄懂 Train (70%), Validation (10%), Test (20%) 各自的神聖不可侵犯性。
3. **掌握超參數微調 (Hyperparameter Tuning)**：了解 Validation Set 在你的雙 GPU 腳本中扮演的「選美裁判」角色。

---

## 2. Deep Dive (核心概念與深度解析)

在統計學習理論 (Statistical Learning Theory) 中，我們假設資料來自於某個未知的真實分佈 $\mathcal{P}(x, y)$。
我們收集到的資料集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N \sim \mathcal{P}$ 只是這個真實母體的一個樣本集合。

我們優化的目標函數（經驗風險最小化 ERM）：
$$ \hat{\theta} = \arg\min_{\theta} \frac{1}{|D_{train}|} \sum_{i \in D_{train}} \mathcal{L}(y_i, f_\theta(x_i)) $$

如果模型擁有無限的容量（Capacity，例如巨大的 Transformer），它能硬生生記憶下所有的 $\mathcal{D}_{train}$。所以我們需要**泛化誤差 (Generalization Error)** 的客觀估計：
$$ \text{Error}_{test} \approx \frac{1}{|D_{test}|} \sum_{i \in D_{test}} \mathcal{L}(y_i, f_{\hat{\theta}}(x_i)) $$

### 資料污染 (Data Leakage)
最嚴重的學術災難發生在「把 Test Set 拿來引導模型訓練」。如果在訓練過程中，模型（甚至是你自己作為工程師）不小心「看」到了 Test Set 的資訊，即使只有一微米的資訊洩漏，你的模型分數就會虛高 (Optimistically biased)。這就是為什麼在你的專案中，最後的 `test_edges` 只能在所有模型架構、Jaccard 閾值都決定好之後，再跑**唯一的一次**。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要 Validation 分數越高越好，我就一直瘋狂調參數去討好 Validation Set。**
  * *真相*：這會導致模型對 Validation Set 也過擬合！這就是為什麼我們需要三個資料集。如果你調了一百萬次參數讓 Validation 考 100 分，它本質上也是另一種被你「手動死背」的題庫，最後 Test 考出來還是會翻車。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們接下來使用 `sklearn` 套件，寫一段具備亂數種子保護（確保實驗可再現 Reproducibility）的資料切分標準程式。

```python
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def rigorous_data_splitting(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    實作標準的 Train(70%) / Val(15%) / Test(15%) 三重切分法。
    在實際的圖神經網路 (如 Amazon dataset) 中，因為關係到邊 (Edges) 的拓樸結構，
    切分會更難（稱為 Transductive/Inductive Split），但核心精神與此完全一致。
    """
    # 為了展示，設定固定的隨機亂數種子，保證每次跑這段 code 得到的切分都一樣
    # 這是證明你沒有「偽造報告」的最基本要求！
    RANDOM_SEED = 42 
    
    # 1. 第一次切分：把資料切成 Train (70%) 和 暫存集 Temp (30%)
    # stratify=y 可以確保分類問題中，各類別的比例在切分後依然均勻 (非常重要!)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    # 2. 第二次切分：把暫存集 Temp (30%) 平分成 Val (15%) 和 Test (15%)
    # 平分等同於 test_size=0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=RANDOM_SEED,
        stratify=y_temp
    )
    
    total_len = len(X)
    print(f"總資料量: {total_len} 筆")
    print(f"Training Set   (練功): {len(X_train)} 筆 ({len(X_train)/total_len:.0%})")
    print(f"Validation Set (模擬): {len(X_val)} 筆 ({len(X_val)/total_len:.0%})")
    print(f"Test Set       (期末): {len(X_test)} 筆 ({len(X_test)/total_len:.0%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # 生成 1000 筆模擬的特徵資料 (隨機整數)
    dummy_X = np.random.rand(1000, 5) # 1000 筆資料，每筆 5 個特徵
    # 生成 1000 筆二元分類標籤 (0 或 1，像是要預測 這條 Edge 存不存在)
    dummy_y = np.random.randint(0, 2, 1000)
    
    rigorous_data_splitting(dummy_X, dummy_y)
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **圖論中的特殊切分陷阱 (Temporal/Structural Edge Splitting)：** 在你的 Amazon 購買紀錄中，如果你隨機把邊打斷切分，圖的拓樸結構會被破壞（本來是好朋友，被你切成測試集後在訓練階段就變成陌生人）。所以真實世界的 Link Prediction 經常依照「時間順序（買了 A 之後才買 B）」來當作 Training 與 Test 的分界，以防模型穿越時空看到未來。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
你發現模型在 Training Set 上的預測準確率高達 99%，但在 Validation Set 上卻只有 55%。請問發生了什麼現象？面對這種情況，你作為一位工程師，應該採取的兩項策略是什麼？（提示：思考「減少記憶力」或是「看更多書」）。

### Extreme Edge-Case (魔王挑戰)
在有些極度缺乏資料的領域（如罕見疾病的醫療影像，全世界只有 100 張 X 光片），如果再將資料切成 Train/Val/Test，Train 裡面只剩下 70 張照片，神經網路根本學不起來。
請查閱並解釋什麼是 **「K-折交叉驗證 (K-Fold Cross-Validation)」**？它如何利用數學輪替的魔法，讓每一筆微小的資料既能當作訓練，又能當作驗證，從而榨乾少量資料的最後一滴價值？