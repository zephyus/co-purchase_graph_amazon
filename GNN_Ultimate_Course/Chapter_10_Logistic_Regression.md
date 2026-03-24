# 第十章：線性迴歸與邏輯迴歸（用一條線做預測）

## 1. The Intuition (引言與核心靈魂)

如果我們有一個點陣圖，標記了「唸書的時數 (X 軸)」與「期末考的成績 (Y 軸)」。
我們看到這些點大致呈現一個往右上角飆升的趨勢。現在，我希望你能拿出一把直尺，在這些點之間畫出一條「最能代表這個趨勢的直線」。這條線就是 **線性迴歸 (Linear Regression)**。只要畫出這條線，下一次有人告訴你他讀了 10 個小時，你只要對照線上對應的 Y 坐標，就能大膽預測他的分數。

但是，如果我們要預測的不是連續的分數（如 85.5 分），而是「會不會被退學（及格 / 不及格）」這種只有 **0 或是 1** 的結果呢？
如果我們硬用一條沒有極限、會衝上外太空的直線去套用 0 和 1 的問題，當有人讀了 1000 個小時，模型可能會預測他的退學機率為 "-500%" 或 "800%"——這在機率學上是徹底荒謬的。

這時，我們需要把那條剛硬無限的直線，透過一個「魔法扭曲機」把它彎折，將其所有預測值軟禁在 $0$ 到 $1$ 之間這條狹小的走廊裡。這台魔法機器就叫做 **Sigmoid 函數**。而被加上了 Sigmoid 的線性迴歸，就華麗變身成為了最強的分類器：**邏輯迴歸 (Logistic Regression)**。這也是深度神經網路最基礎的一塊積木。

### Learning Objectives (學習目標)
1. **掌握線性迴歸的幾何意義**：理解權重 (Weights) 與偏誤 (Bias) 在方程式中扮演的角色。
2. **精通 Sigmoid 函數**：明白我們為什麼必須要把數字「擠壓」到 $[0, 1]$ 之間。
3. **理解 Logits 的概念**：為你的下一個階段（PyTorch 與 `BCEWithLogitsLoss`）鋪平道路，這是讓你的程式免於爆炸的護城河。

---

## 2. Deep Dive (核心概念與深度解析)

在機器學習的舞台上，最優美的雙人舞就是線性與非線性 (Linear & Non-linear)。

### 1. 矩陣形式的線性迴歸
假設我們有 $d$ 個特徵，構成特徵向量 $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$。
模型的大腦裡有一組權重向量 $\mathbf{w}$ 與偏置項 $b$。
模型的預測等同於求它們的內積 (Dot Product)：
$$ z = \mathbf{w}^T \mathbf{x} + b $$
這個 $z$ 在深度學習的術語中被稱為 **Logit**（未經過激勵的原始分數）。$z$ 的值域可以是 $(-\infty, \infty)$。

### 2. 邏輯迴歸 (Logistic Regression) 與 Sigmoid
為了執行二元分類（如圖論中判斷 Link Prediction：邊是否存在 1 或 0），我們必須將這個無限大的 $z$ 轉換成一個合法的機率分佈 $p \in [0, 1]$。
我們引入 **Sigmoid 函數 $\sigma(z)$**：
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
當 $z$ 是一個極大的正數（例如 $z=100$），$e^{-100}$ 趨近於 0，$\sigma(100) \approx 1$ (模型極度自信是 1)。
當 $z$ 是一個極大的負數（例如 $z=-100$），分母會變得無限大，$\sigma(-100) \approx 0$ (模型極度自信是 0)。
當 $z = 0$ 時，$\sigma(0) = \frac{1}{1+1} = 0.5$ (模型對結果五五波，處於猶豫不決的最邊界)。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：邏輯迴歸裡面有「迴歸」兩個字，所以它是用來連續預測數字的。**
  * *真相*：這是歷史遺留的命名烏龍。邏輯迴歸是血統純正的**分類 (Classification) 演算法**，它的用途是區分 0 和 1（或是多類別），而不是用來預測像房價或氣溫這種無限連續的數值。

---

## 3. Code & Engineering (程式碼實作與工程解密)

這段程式碼我們不用任何套件幫忙，從零手刻一個邏輯迴歸的純數學前向傳播。同時展示什麼是讓超級電腦都會崩潰的數值不穩定 (Numerical Instability)。

```python
import numpy as np

def sigmoid(z: float) -> float:
    """計算 Sigmoid 函數。它會將任何實數擠壓到 (0, 1) 之間。"""
    return 1.0 / (1.0 + np.exp(-z))

def simulate_logistic_regression() -> None:
    """
    透過點積計算 Logits，並透過 Sigmoid 得到機率。
    同時展示機器學習中最怕遇到的浮點數爆炸。
    """
    
    # 假設這是一個圖論節點上的 3 個特徵
    features_x = np.array([1.5, -2.0, 3.1])
    
    # 模型透過訓練後，學會的 3 個權重加上 1 個偏誤
    weights_w = np.array([0.8, -1.2, 0.5])
    bias_b = 0.1
    
    print("--- 步驟 1: 計算基礎線性模型 (Logit z) ---")
    # np.dot 就是拿 [1.5*0.8 + (-2.0)*(-1.2) + 3.1*0.5]
    logit_z = np.dot(weights_w, features_x) + bias_b
    print(f"原始火力分數 z = {logit_z:.4f} (值域無限制，這數字到底算高還算低？)")
    
    print("\n--- 步驟 2: 經過魔法扭曲機 (Sigmoid) 轉換為機率 ---")
    probability_p = sigmoid(logit_z)
    print(f"預測為正類 (有連線) 的機率 p = {probability_p:.4f} ({probability_p*100:.1f}%)")
    
    print("\n--- 🧨 步驟 3: 數值爆炸危機示範 ---")
    # 假設因為梯度爆炸，模型的權重變成了不可思議的極端數字
    crazy_z = 2000.0  # e^(-2000) 會在電腦中引發極限下溢出
    # 在純 Python / Numpy 中，這可能會跳出 RuntimeWarning: overflow encountered in exp
    try:
        crazy_p = sigmoid(crazy_z)
        print(f"當 logit_z = {crazy_z} 時，Sigmoid 回傳: {crazy_p}")
    except Exception as e:
        print(f"程式崩潰啦！{e}")
    # 工程註解：這就是為什麼在你的 Amazon 專案中，我們要寫 `logits = torch.clamp(logits, min=-20.0, max=20.0)` 的原因！

if __name__ == "__main__":
    simulate_logistic_regression()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **Logits 的極限夾鉗 (Clamping / Clipping)：** 當 $z \to \infty$ 時，$e^{-z}$ 在 IEEE 754 浮點數表示法中可能會變成非正規數甚至 `0.0`。如果在反向傳播時我們要把預測機率拿去取 $\log(p)$ 來計算 BCE 損失，一旦 $p=0.0$，$\log(0)$ 在數學上是負無限大 ($-\infty$)，這就會變成程式裡的 `NaN` (Not a Number)。`NaN` 就像是癌細胞，只要一個神經元變成 `NaN`，整個神經網路的權重在下一個 Epoch 就會全部變成 `NaN`！解法就是我們在第七階段 (作業實戰) 時加上去的**防火牆**：夾緊 logits。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
Sigmoid 函數 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 有一個非常美麗的微積分恆等式。請查閱微積分的連鎖律 (Chain Rule)，並證明對 Sigmoid 函數取一次微分 (Derivative) 的結果，可以被優雅地寫成：
$$ \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) $$
這為什麼對早期的第一代神經網路在計算背向傳播 (Backpropagation) 帶來了巨大的效能幫助？

### Extreme Edge-Case (魔王挑戰)
仔細觀察上面微積分證明的結果：$\sigma'(z) = p \cdot (1 - p)$，其中 $p$ 是預測的機率。
如果神經網路處於極端自信的狀態，比如說它預測 $p = 0.9999$ 或是 $p = 0.0001$。
請把這個極端的 $p$ 帶入微分公式，算算看算出來的梯度向量 $\sigma'(z)$ 等於多少？
這數字幾乎等於零！如果梯度變成了 0，模型的學習就會完全停滯。這就是惡名昭彰的 **「梯度消失 (Vanishing Gradient) 問題」**。請簡述這個數學缺陷，為何讓當代深度大模型 (如 ChatGPT 或大型 GNN) 必須拋棄 Sigmoid 隱藏層，而轉投向 ReLU (Rectified Linear Unit) 的懷抱？