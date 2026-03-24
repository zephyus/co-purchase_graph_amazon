# 第九章：為什麼你作業需要 F1-Score 與 ROC-AUC？（不平衡資料的救星）

## 1. The Intuition (引言與核心靈魂)

接續著上一章那要命的「白痴 AI 醫生」，如果準確率 (Accuracy) 不管用，那我們該拿什麼來逼迫模型成為一位真正的神醫呢？這就需要引進兩個在工業界（以及你的 Amazon 作業中）真正掌握生殺大權的指標：**Precision (精確率)** 與 **Recall (召回率)**。

*   **Precision (精緻的獵手)**：當 AI 指著一張 X 光片說：「這人有病！」時，它說中的機率有多高？（抓出來的人裡面，多少是真犯人？）
*   **Recall (狂野的漁網)**：世界上所有真的有病的人中，AI 到底成功「撈」出了幾個？有沒有漏網之魚？

如果把這兩個指標合體，我們會得到一個終極的平衡分數 **F1-Score**；而如果我們想觀察模型在「各種不同信心門檻下」的綜合實力，我們就會看 **ROC-AUC**。這就是你的期末專案最後嚴格要求的雙基準。

### Learning Objectives (學習目標)
1. **掌握 P-R 拔河遊戲**：理解為什麼 Precision 和 Recall 天生就是互相抵觸的敵人。
2. **計算調和平均 (Harmonic Mean)**：了解為什麼 F1-Score 不是用普通的算術平均算出來的。
3. **理解曲線下的面積 (ROC-AUC)**：看懂模型預測機率的排序能力，而不是僅限於 0/1 的絕對分類。

---

## 2. Deep Dive (核心概念與深度解析)

讓我們把混淆矩陣的四個金剛 (TP, FP, TN, FN) 用於建立更強大的武器。

### 1. Precision 與 Recall 的數學定義
$$ \text{Precision (精確率)} = \frac{TP}{TP + FP} $$
$$ \text{Recall (召回率)} = \frac{TP}{TP + FN} $$

**P-R 的拔河效應**：如果我們下令「不准放過任何一個壞人」，模型就會把所有人都當成壞人 (Threshold = 0.0) -> Recall 變成 100%，但 Precision 會暴跌為接近 0 (因為抓了超多好人 FP)。反之亦然。

### 2. F1-Score：平衡的藝術
為了同時兼顧 Precision 與 Recall，我們取它們的**調和平均數 (Harmonic Mean)**，而不是算術平均數。
$$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$
**為什麼不用算術平均 $(P+R)/2$？** 如果一個模型的 P=1.0, R=0.0，算術平均會是 0.5 (看起來還行)，但如果是調和平均，只要其中一者是 0，整個 F1-Score 就會被暴力往下扯，變成 0。這迫使模型**不能偏科**，必須兩者兼得才能拿到高 F1。在你作業的 Threshold 搜索中，我們就是沿著閥值找到 F1 的最高峰。

### 3. ROC-AUC：不要門檻，只要實力
ROC 曲線 (Receiver Operating Characteristic) 的 X 軸是 FPR (False Positive Rate，錯殺無辜率)，Y 軸是 TPR (就是 Recall)。
AUC (Area Under Curve) 代表 ROC 曲線下的面積。
*  AUC = 0.5：像丟硬幣一樣完全瞎猜。
*  AUC = 1.0：完美神明，把所有正樣本的機率都排在所有負樣本前面。
AUC 偉大之處在於，它**不需要你先決定 0.5 這個閾值**，它衡量的是模型把你輸出的概率「由大到小排序」的絕對實力！

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要我一直調整 threshold，就可以把 ROC-AUC 調高。**
  * *真相*：Threshold 的調整**完全不會**改變 ROC-AUC，只會改變你在 ROC 曲線上的「位置」（也就是會改變 F1, P, R）。AUC 是整個模型預測機率分佈的根本品質，代表了特徵提取（如 GNN 嵌入）的根本能力。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們來寫一段實務上每天都在使用的評估函數。我們直接使用 `sklearn.metrics` 這個工業標準庫來印出你的期末成績單。

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model_metrics() -> None:
    """
    展示如何從模型的「機率預測值 (Probabilities)」
    轉換成二元預測，並計算 F1 與 ROC-AUC。
    """
    
    # 模擬 10 個測試樣本
    # Truth: 只有 3 個人有病 (1)
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    
    # 模型輸出的機率 (還沒被二元化前)
    # y_probs: GNN 經過 sigmoid 後的輸出空間 [0, 1]
    y_probs = np.array([0.1, 0.4, 0.9, 0.2, 0.8, 0.6, 0.3, 0.05, 0.45, 0.4])
    
    # 1. 計算 ROC-AUC (只需要真實類別與預測機率，不需要閾值！)
    # 這是衡量 GNN 把 "有連線" 的配對打高分的「純天然能力」
    auc_score = roc_auc_score(y_true, y_probs)
    print(f"🌟 模型天然排序能力 (ROC-AUC): {auc_score:.4f}")
    
    # 2. 測試不同的閾值 (Threshold) 對 F1 的影響
    thresholds = [0.3, 0.5, 0.7]
    print("\n--- 尋找最佳 F1 的旅程 (Threshold Search) ---")
    
    for th in thresholds:
        # NumPy 布林遮罩自動轉為 0 或 1 (如 y_probs 裡 0.9 > 0.5 就變為 True 即 1)
        y_pred = (y_probs >= th).astype(int)
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"當閾值為 {th:.1f} 時:")
        print(f"  -> Precision: {p:.2f} | Recall: {r:.2f} | 🎯 F1: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model_metrics()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **尋找最佳驗證閾值 (Validation Threshold Search)：** 在你的 Q4 程式中，我們在 Validation Set 上跑了一個極其細密的迴圈（掃過 `thresholds = np.linspace(0.1, 0.9, 100)`），尋找讓 F1 最高的那個魔法門檻（例如發現這波資料的最佳切割線其實是 0.3 而不是預設的 0.5）。這就是為什麼你的專案最終能一舉突破 85% F1 Baseline 的致勝關鍵！

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
考慮以下極端情況：有一個檢測炸彈的模型。如果包裹裡有炸彈，模型漏掉沒響，會導致整個機場炸毀；如果包裹裡沒炸彈，但模型誤響了，只會導致旅客多花 5 分鐘重檢。
在這種極端的「代價不對稱」中，身為工程總監，你會刻意去偏袒 Precision 還是 Recall？請說明為什麼。

### Extreme Edge-Case (魔王挑戰)
如果我們有一個極端糟糕的模型，它的預測機率和真實情況存在著完美的**反向關係**。它總是把真的壞人給予 0.05 的極低機率，把好人給予 0.99 的極高機率。
1. 請問它預測的 ROC-AUC 分數會非常接近多少（0.0 還是 0.5）？
2. 在工程上，如果你遇到拿到這種極端分數的模型，你不需要重新訓練它，只需在程式碼中加上「一行簡單的操作」就能讓它瞬間變成世界第一的神明模型。請問這一步微小但暴力的神偷操作是什麼？