# 第八章：打分數的標準（準確度 Accuracy 是什麼？）

## 1. The Intuition (引言與核心靈魂)

如果在一場英文考試中，你隨便用丟硬幣的方式作答選擇題，結果滿分 100 分你拿了 25 分。老師跟你說：「你的準確率 (Accuracy) 是 25%」。這非常直觀。
這就是分類問題中最基礎的指標：**你猜對了幾題，除以總題數**。

但現在想像一個生死的醫療場景：有一種極度罕見的疾病，每一萬個人之中只有一個人會得病。
如果我寫了一個「白痴 AI 醫生」，它的程式碼只有一行：`def predict(patient): return "沒病"`。
這個 AI 醫生預測了一萬個人，猜對了 9999 個人（因為他們本來就沒病），只有那個真正得病的人被它害死了。
計算它的準確率：$9999 / 10000 = 99.99\%$。

一個會害死人的廢物模型，居然獲得了 99.99% 的超神級分數！
這就是機器學習入門者最常踩的坑：**「不平衡資料 (Imbalanced Data)」下的準確率幻覺**。在這一章，我們要把純粹的 Accuracy 拆解開來，看見裡面隱藏的陷阱。

### Learning Objectives (學習目標)
1. **理解 Accuracy 的局限**：知道什麼時候可以用它，什麼時候絕對不能用它。
2. **拆解混淆矩陣 (Confusion Matrix)**：學會解讀 TP, TN, FP, FN 四大金剛。
3. **區分兩大犯罪類別**：「寧可錯殺一百 (False Positive)」與「不可放過一個 (False Negative)」。

---

## 2. Deep Dive (核心概念與深度解析)

在二元分類問題 (Binary Classification) 中，正確答案 $y \in \{0, 1\}$，模型預測 $\hat{y} \in \{0, 1\}$。通常 $1$ 代表「正類/陽性/目標事件發生」，$0$ 代表「負類/陰性/事件未發生」。

### 1. Accuracy (準確率) 的數學定義
$$ \text{Accuracy} = \frac{\text{所有預測正確的數量}}{\text{總樣本數}} = \frac{TP + TN}{TP + TN + FP + FN} $$

要深刻理解分母裡的四個怪獸，我們必須引進 **混淆矩陣 (Confusion Matrix)** 的概念。
*   **True Positive (TP / 真陽性)**：AI 說有病 (1)，真實也是有病 (1)。英雄！
*   **True Negative (TN / 真陰性)**：AI 說沒病 (0)，真實也是沒病 (0)。安全！
*   **False Positive (FP / 偽陽性 / 誤報型犯罪 / Type I Error)**：AI 說有病 (1)，但其實沒病 (0)。就像健康的人被宣告得癌症，虛驚一場；或是警報器亂響。
*   **False Negative (FN / 偽陰性 / 漏報型犯罪 / Type II Error)**：AI 說沒病 (0)，但其實有病 (1)。這是最要命的，有病沒被查出來，會直接導致死亡。

當你面對「高度不平衡 (Highly Imbalanced)」的資料時（例如你的 Amazon 網路中，兩個隨便的路人之間『沒有連線=0』的機率是 99.99%），如果你的模型變成一個只會喊 0 的白痴，它的 TN 會極度巨大，掩蓋了極度幼小的 TP 與 FN，從而製造出高 Accuracy 的假象。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要 Accuracy 很高，這個 AI 就可以上線賣錢了。**
  * *真相*：在銀行抓詐騙、醫院找異常、工業找瑕疵，乃至圖論的 Link Prediction 領域，Accuracy 幾乎是一張「廢紙」。我們需要的是下一章會教的武器：Recall、Precision、F1 與 ROC-AUC。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們用程式碼來親眼見證那個 99% 的分數是如何被「白痴演算法」騙出來的。

```python
import numpy as np

def demonstrate_accuracy_illusion() -> None:
    """
    這個函數揭示了在『極端不平衡資料集』下，
    為什麼單純看 Accuracy 是一場災難。
    """
    
    # 模擬 1000 個病人。990 個健康 (0)，10 個得罕見疾病 (1)
    y_true = np.zeros(1000)
    y_true[:10] = 1 
    np.random.shuffle(y_true) # 打亂順序
    
    # 建構一個「白痴 AI」: 不管輸入什麼，一律預測 0 (沒病)
    y_pred_idiot = np.zeros(1000)
    
    # 數學計算 TP, TN, FP, FN
    # (y_true == 1) & (y_pred == 1) 代表兩者皆為 1
    TP = np.sum((y_true == 1) & (y_pred_idiot == 1))
    TN = np.sum((y_true == 0) & (y_pred_idiot == 0))
    FP = np.sum((y_true == 0) & (y_pred_idiot == 1))
    FN = np.sum((y_true == 1) & (y_pred_idiot == 0))
    
    accuracy = (TP + TN) / len(y_true)
    
    print("--- 🩺 白痴 AI 醫生的績效報告 ---")
    print(f"真實病人數量: {np.sum(y_true)}")
    print(f"機器找出的病人 (TP): {TP}")
    print(f"機器漏掉的病人 (FN): {FN}  <-- 【重大醫療疏失！】")
    print("-" * 30)
    print(f"🏆 模型準確率 (Accuracy): {accuracy * 100:.2f}%")
    print("\n結論：拿到 99.00% 高分的 AI，卻害死了所有(10個)病患。")

if __name__ == "__main__":
    demonstrate_accuracy_illusion()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **不平衡圖論的抽樣技術 (Negative Sampling)：** 在你的作業裡，真正有連線 `edge=1` (正類) 的配對遠遠少於沒有連線 `edge=0` (負類) 的配對。如果不做處理去算 BCE Loss，模型就會退化成上述的「白痴 AI」。因此，我們在抽樣 `val_edges` 時，通常會故意抽取 **1:1** 比例的正負邊。這稱為負抽樣 (Negative Sampling)，是拯救圖神經網路不被淹沒的關鍵技巧。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
有兩家公司在推銷他們的「電子郵件垃圾信攔截 AI」。
- 第一家 AI 的特性是：**極高的 False Positive**，它很敏感，一看到稍微可疑的信就當作垃圾信丟到垃圾桶中。
- 第二家 AI 的特性是：**極高的 False Negative**，它很遲鈍，除非 100% 確定是垃圾信，否則都會放行到你的主要信箱。
作為一個每天等著收「重要面試通知」的求職者，你會買哪一家的 AI？請用 FP 與 FN 的代價 (Cost) 來解釋你的決策。

### Extreme Edge-Case (魔王挑戰)
如果我們不使用預測類別（0或1），而是輸出一個機率 $p \in [0, 1]$（例如這個病人得病的機率是 0.8）。
我們規定當 $p > \text{Threshold}$ 時判斷為 1，否則為 0。
請思考：如果我們把 Threshold 從 0.5 慢慢調高到 0.99，這時神經網路必須「極端自信」才會說人有病。這會讓 False Positive (誤報) 與 False Negative (漏報) 分別產生什麼方向的變化？這對理解下一章的 F1 和 ROC-AUC 有決定性的幫助。