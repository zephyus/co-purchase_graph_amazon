# 第十一章：模仿大腦的網路（神經網路 Neural Networks 基礎）

## 1. The Intuition (引言與核心靈魂)

在第十章，我們學到了「邏輯迴歸」：輸入特徵，乘上相對應的權重，塞進一個 Sigmoid 擠壓成機率。
如果邏輯迴歸是一條線（一個決策平面），那麼當你的資料就像是一個「太極圖」（一半黑一半白，互相環繞交錯在同一個圓內）的時候，你無論怎麼用直尺畫線，都會把黑和白切錯。因為這世界大部分的規律（如語言、圖像、圖論拓樸）絕對**不是線性的**。

如果一個神經元（Logistic Regression）做不到，那我們就用一萬個神經元！
我們把神經元排成一層一層的。第一層神經元負責抓取「這條線是斜的還是直的」，第二層負責把這些線條組合成「這個圖形有耳朵和尾巴的形狀」，最後一層神經元（負責投票）總結前幾層的意見說：「這是一隻貓！」
這就是 **深度學習神經網路 (Deep Neural Networks, DNN)** 的本質：透過**多層次的非線性特徵轉換 (Hierarchical Non-linear Feature Transformation)**，去逼近宇宙中任何一種極端複雜的數學函數。

### Learning Objectives (學習目標)
1. **理解隱藏層 (Hidden Layers)**：知道為什麼網路需要「深 (Deep)」。
2. **掌握激勵函數 (Activation Functions)**：從 Sigmoid 升級到 ReLU，理解打破線性的必要性。
3. **認識 MLP (多層感知機)**：圖神經網路 (GAT/GCN) 裡面用來做最終預測的本體裝備。

---

## 2. Deep Dive (核心概念與深度解析)

從數學上看，一個具有單一隱藏層的神經網路可以被寫成兩個函數的複合 (Composition)：
$$ \mathbf{h} = \sigma_1(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) $$
$$ \mathbf{\hat{y}} = \sigma_2(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}) $$

*   $\mathbf{x} \in \mathbb{R}^{d_{in}}$：輸入層向量。（例如你的 Graph node feature 是 128 維）。
*   $\mathbf{W}^{(1)} \in \mathbb{R}^{d_{hid} \times d_{in}}$：第一層的權重**矩陣 (Matrix)**。從一個神經元進化到了矩陣運算！
*   $\mathbf{h} \in \mathbb{R}^{d_{hid}}$：隱藏層向量，這是經過第一次非線性扭曲後的「高級特徵」。
*   $\sigma_1$：隱藏層的激勵函數，現代幾乎統一使用 **ReLU (Rectified Linear Unit)**：$f(x) = \max(0, x)$。

### 為什麼不可以沒有激勵函數？ (The Importance of Non-linearity)
這是神經網路最經典的考題。如果我們把 $\sigma_1$ 和 $\sigma_2$ 拿掉，網路會變成這樣：
$$ \mathbf{\hat{y}} = \mathbf{W}^{(2)} (\mathbf{W}^{(1)} \mathbf{x}) = (\mathbf{W}^{(2)} \mathbf{W}^{(1)}) \mathbf{x} = \mathbf{W}_{new} \mathbf{x} $$
你看出來了嗎？兩個矩陣相乘 $\mathbf{W}^{(2)} \mathbf{W}^{(1)}$ 在數學上可以被摺疊成一個新的矩陣 $\mathbf{W}_{new}$。
如果你不用非線性函數在每一層中間當作「隔板」，那麼不管你疊了 100 層還是 1000 層，**它最終都只會退化倒縮成一個巨大且無趣的線性迴歸**。它依舊解決不了太極圖的問題。這被稱為 Universal Approximation Theorem (通用近似定理) 的破滅。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：神經網路越深越深層，它就一定越聰明。**
  * *真相*：這只是科幻片的幻想。越深的神經網路越容易遭遇「梯度消失」或「特徵過度平滑 (Over-smoothing)」。在 GNN 裡面（尤其是你的作業），如果 GNN 疊超過 4 層，所有節點的特徵會糊成一團，導致不管哪個節點看起來都一模一樣，這被稱為 GNN 界死神等級的 Over-smoothing 問題。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們現在要告別 NumPy 手刻的黑暗時代，正式踏入工業標準武器庫：**PyTorch**。這段示範是神經網路最基礎的前向傳播堆疊。

```python
import torch # PyTorch 機器學習的霸主
import torch.nn as nn
import torch.nn.functional as F

# 在 PyTorch 中，所有的自訂網路都必須繼承 nn.Module
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化神經網路的所有大腦零件 (權重矩陣)。
        """
        super(SimpleMLP, self).__init__() # 呼叫父類別初始化
        
        # 定義第一層 (線性轉換層 Linear Transformation): W1 * x + b1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # 定義第二層: W2 * h + b2
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定義資料如何從輸入端流向輸出端 (Forward Pass)。
        """
        # 第一步：穿過第一層權重
        out = self.fc1(x)
        
        # 第二步：非常重要！加入 ReLU 激勵函數，打破線性，賦予網路靈魂
        out = F.relu(out)
        
        # 第三步：穿過輸出層。
        # 注意：我們通常不在網路的最後加上 Sigmoid。
        # 為什麼？因為 PyTorch 有一個叫 BCEWithLogitsLoss 的神兵利器，
        # 它可以把 Sigmoid 和 Loss 融合在底層 C++ 用更安全的數學(Log-Sum-Exp)一次算完，防爆炸！
        logits = self.fc2(out)
        
        return logits

def test_mlp_forward() -> None:
    # 建立一個大腦模型：輸入特徵 128 維，隱藏層 64 維，輸出 1 維 (預測用)
    model = SimpleMLP(input_dim=128, hidden_dim=64, output_dim=1)
    
    # 印出模型的架構，你會看到在神經科學與工程學中完美結合的結構
    print("--- 🧠 構建的神經網路結構 ---")
    print(model)
    
    # 模擬 5 筆資料 (Batch Size = 5)，每筆 128 個特徵
    # 在神經網路中，所有資料都必須轉為 PyTorch Tensor 格式
    dummy_x = torch.randn(5, 128) 
    
    # 把資料丟入模型 (會自動呼叫 forward 函式)
    raw_logits = model(dummy_x)
    
    print("\n--- ⚡ 網路產生的火力分數 (Logits) ---")
    print(raw_logits)
    print("注意這 5 個數字可以大於 1 也可以是負的，因為它們還沒被 Sigmoid 塞進去。")

if __name__ == "__main__":
    test_mlp_forward()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **批量運算 (Batching)：** 在深度學習中，我們**絕對不會**一次把單一筆資料送進模型（`x.size() = [128]`）。我們會把幾千筆資料捆綁成一個 Batch 矩陣（`x.size() = [2048, 128]`）。之所以這麼做，是因為 GPU 的幾萬個平行運算核心極度飢渴。如果你一次只餵一筆，GPU 效能利用率不到 1%，就像你雇用了十萬大軍卻讓他們排隊一個接一個運磚頭一樣浪費。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
ReLU 的函數定義是 $\max(0, x)$。如果 $x \le 0$，那導數 (Gradient) 就是 0。
如果在訓練初期的某一瞬間，某個隱藏層的神經元它前面的權重很不巧地導致了它收到的輸入全部都是負數。
請解釋這個神經元會發生什麼事？這在深度學習中為何被稱為 **「Dead ReLU Problem (神經元死亡問題)」**？

### Extreme Edge-Case (魔王挑戰)
在上述程式碼中，`self.fc1` 是一個 `[128, 64]` 的線性層；`self.fc2` 是一個 `[64, 1]` 的線性層。考慮加上偏差值 (Bias)。
請展現你的算力，手動計算出這整個神經網路**總共有多少個浮點數參數 (Learnable Parameters)**？
(提示：權重矩陣參數 + Bias 向量參數。計算出確切的整數數字。這種算力推估是你決定買 VRAM 24G 還是 80G 顯卡的最核心基礎能力。)