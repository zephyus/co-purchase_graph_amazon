# 第六章：什麼是機器學習？（人類經驗 vs. 機器找規律）

## 1. The Intuition (引言與核心靈魂)

想像你是一家手搖飲料店的老闆。過去十年來，你每天都在觀察天氣和顧客的點單。你腦中有一本「隱形字典」：如果是大晴天且氣溫超過 35 度，冰塊要稍微多一點；如果是下雨天，熱飲和少冰的比例會上升。這種基於歷史經驗做出的直覺判斷，就是「人類學習」。

傳統的程式設計（如前五章所學）是**我們直接把規則寫死給電腦執行**。我們寫下：`if temperature > 35 and weather == "Sun": serve("多多冰")`。
但在現實中，影響飲料銷售的變數太多了（氣溫、濕度、是否有打折、旁邊有沒有開新店...），你根本不可能列出所有的 `if/else` 條件。

**機器學習 (Machine Learning)** 翻轉了這個範式。我們不再給電腦「規則」，而是直接把過去十年的「歷史資料（氣溫、天氣、最終賣了幾杯）」丟給電腦，並給它一個數學框架，讓它自己去**「算」**出那條隱形的規則。當未來出現一個沒見過的天氣時，它就能用自己找出的規則來做預測。

### Learning Objectives (學習目標)
1. **翻轉編程範式**：理解 Rule-based Programming 與 Machine Learning 的本質差異。
2. **掌握 ML 三大要素**：資料 (Data)、模型 (Model)、損失函數 (Loss Function)。
3. **認識監督式學習 (Supervised Learning)**：這是解答你 Amazon 專案的核心大類。

---

## 2. Deep Dive (核心概念與深度解析)

機器學習的核心可以用一個極簡的數學等式來概括：
$$ \hat{y} = f_\theta(x) $$

* **$x$ (Features / 特徵)**：我們已知的輸入資料。例如節點特徵、氣溫、商品價格。在數學上通常表示為一個向量 (Vector) $\mathbf{x} \in \mathbb{R}^d$。
* **$y$ (Labels / 標籤)**：我們真正想預測的答案（Ground Truth）。如果這是一場考試，$x$ 是題目，$y$ 就是標準答案。
* **$f_\theta$ (Model / 模型)**：電腦大腦裡的那個函式。$\theta$（Theta）代表「模型的內部參數（如圖神經網路裡的權重）」。
* **$\hat{y}$ (Prediction / 預測值)**：模型目前猜測的答案。讀作 "y-hat"。

### 機器的「學習」是怎麼發生的？
學習的過程，本質上是一個**最佳化問題 (Optimization Problem)**。
我們定義一個**損失函數 $\mathcal{L}(y, \hat{y})$**，用來衡量「標準答案」與「預測值」之間的差距（Loss）。差距越大，表示模型越笨。
機器的任務就是透過演算法，不斷微調自己腦中的參數 $\theta$，使得總損失最小化：
$$ \theta^* = \arg\min_{\theta} \sum_{i=1}^{N} \mathcal{L}(y_i, f_\theta(x_i)) $$
這就是所有 AI（包含 ChatGPT 與你的 Amazon GNN）的最底層世界觀。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：機器學習模型像人腦一樣，真的「看懂」了圖片或文字。**
  * *真相*：機器學習模型只懂數字（矩陣與張量）。它並不知道一張貓的圖片長怎樣，它只是找到了一種複雜的超高維度幾何變換，把這些像素的數字組合投影到了一個被標記為「貓」的數學空間座標裡。

---

## 3. Code & Engineering (程式碼實作與工程解密)

這段程式碼將展示一個最原始的「線性迴歸」概念：給定 $x$，讓機器去猜出隱藏的 $\theta$（即斜率與截距），來逼近 $y$。這也是所有深度學習網路 (包含 GNN 中的線性層) 的老祖宗。

```python
import numpy as np

def simulate_machine_learning() -> None:
    """
    透過一個超簡化的「猜數字」迴圈，展示機器學習中
    前向傳播 (Forward Pass) 與 誤差計算 (Calculate Loss) 的思想。
    """
    
    # 1. 準備 Data (這就是大自然的真理，但機器目前不知道)
    # 真實邏輯是: y = 3 * x + 2 (斜率為3, 截距為2)
    x_data = np.array([1, 2, 3, 4, 5])
    y_true = np.array([5, 8, 11, 14, 17])
    
    # 2. 定義 Model (機器的大腦起初是一片空白，隨機瞎猜)
    # 我們假設神經網路只有一個大腦細胞 (參數)，叫做 theta (斜率)
    # 先忽略截距，讓機器去學出最好的斜率
    theta_guess = 1.0  # 機器一開始瞎猜斜率是 1
    
    # 3. Training Loop (機器的學習週期)
    learning_rate = 0.05
    epochs = 40
    
    print("--- 啟動機器學習訓練 ---")
    for epoch in range(epochs):
        # Forward Pass (預測): y_hat = theta * x
        y_pred = theta_guess * x_data
        
        # Calculate Loss (計算誤差): 使用 Mean Squared Error (MSE)
        # 為什麼要平方？因為差異可能是負的，平方可以消除正負符號並放大巨大誤差
        error = y_pred - y_true
        loss = np.mean(error ** 2)
        
        # Backward Pass / Update (微調大腦)
        # 這裡不使用微積分，而是用直觀的梯度下降概念：
        # 如果 error 是正的 (猜太高)，斜率應該調低；如果是負的 (猜太低)，斜率調高
        gradient = np.mean(error * x_data) 
        theta_guess = theta_guess - learning_rate * gradient
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | 誤差(Loss): {loss:7.4f} | 機器學到的模型參數 theta: {theta_guess:.4f}")

    print(f"\n最終機器認為的公式大約是: y = {theta_guess:.2f} * x")
    print("(雖然正確答案還需要截距+2，但機器已經很努力拼湊出接近 3.x 的斜率網路了！)")

if __name__ == "__main__":
    simulate_machine_learning()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **特徵縮放 (Feature Scaling)：** 如果 $x_1$ 代表身高 (180 cm)，而 $x_2$ 代表細胞大小 (0.0001 cm)，機器在訓練（梯度下降）時，會嚴重偏袒數字看起來很大的 $x_1$。工程上，我們**必須**在資料進模型前，做標準化 (Standardization，即減去平均值、除以標準差 $\frac{x-\mu}{\sigma}$)，否則模型永遠無法收斂。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
在機器學習中，如果一個模型對訓練資料的預測 Loss 降到了 0.0000，完美答對了所有歷史資料，這是否代表這是一個完美的好模型？提示：請搜尋並解釋「過擬合 (Overfitting)」的概念。

### Extreme Edge-Case (魔王挑戰)
在剛剛的程式碼中，`loss = np.mean(error ** 2)` 使用了均方誤差 (MSE)。
如果資料中有一筆「極端離群值 (Outlier)」，例如某天飲料因為有人包場訂了 1000 杯 ($y_{outlier} = 1000$)。這筆資料被平方後，產生的誤差會巨大到把整條迴歸線拉偏。
請查閱文獻，在面對帶有大量雜訊與離群值的資料時，為什麼業界經常使用 **MAE (Mean Absolute Error, $|y - \hat{y}|$ )** 或是 **Huber Loss** 來取代 MSE？請從數學導數 (Gradient) 的穩定性來解釋。