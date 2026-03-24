# 第十三章：神經網路如何學習？（梯度、倒傳遞與優化器 Adam）

## 1. The Intuition (引言與核心靈魂)

想像你被蒙上眼睛（你不知道最佳解答在哪裡），空降在一座高低起伏的群山之中（Loss 崇山峻嶺）。你的唯一目標是走到這座山谷的最低點（Loss = 0，代表模型完美）。因為你看不到，你只能用腳去感受四周：哪邊的地勢斜往下，你就朝哪個方向走一步。
這個用腳感知斜率的動作，就叫做 **計算梯度 (Calculating Gradients)**。這一步的大小，叫做 **學習率 (Learning Rate)**。這個蒙眼下山的過程，就是整個深度學習的核心：**梯度下降法 (Gradient Descent)**。

可是，你的神經網路（大腦）有上百萬個神經元（參數 $\theta$），要怎麼一次同時知道這一百萬個方向的傾斜度？這就是 **反向傳播演算法 (Backpropagation)** 發揮神蹟的時刻。當你走到山谷底時，恭喜你，你的網路宣佈「收斂 (Converged)」，它已經從資料中學會了規律！

### Learning Objectives (學習目標)
1. **理解微積分鏈鎖律的威力**：明白反向傳播 (Backprop) 為什麼被譽為近代 AI 最重要的演算法突破。
2. **掌握神聖的三聯擊**：熟悉 PyTorch 中永遠綁在一起的三行代碼：`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`。
3. **認識優化器之王 (Adam)**：了解為什麼我們不用最單純的 SGD，而必須用帶有「動量 (Momentum)」的 Adam。

---

## 2. Deep Dive (核心概念與深度解析)

設我們有一個擁有兩個權重的極簡神經網路 $y = w_2(w_1 x)$。
假設我們的 Loss 函數為 $\mathcal{L}$，那模型大腦想要學的就是：「我現在把 $w_1$ 扭大一點點，Loss 會跟著變大還是變小？」也就是要求偏微分 $\frac{\partial \mathcal{L}}{\partial w_1}$。

### 反向傳播 (Backpropagation) 的數學原理
根據大一微積分的鏈鎖律 (Chain Rule)，我們不需要硬去求那條複雜公式的直接微分，我們可以把它拆解成「從最後一層往回推」的碎積木相乘：
$$ \frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2}(\text{這裡不用等我}) \dots \cdot \frac{\partial (w_2 w_1 x)}{\partial w_1} $$
這就像是公司裡的問責制度：大老闆 (Loss) 發現賠錢了，他罵部門經理 $\frac{\partial \mathcal{L}}{\partial \text{層}}$，部門經理再根據連帶責任往下罵小主管，一路追溯回最源頭的員工 $w_1$。

### 為什麼是 Adam？ (Adaptive Moment Estimation)
最原始的隨機梯度下降 (SGD) 指令是：
$$ \theta_{new} = \theta_{old} - \alpha \nabla_{\theta}\mathcal{L} $$
（$\alpha$ 是學習率）
但如果是單純的 SGD，一旦遇到一個平緩的山谷地帶，因為梯度 $\nabla \approx 0$，它就會停在那邊走到死。
**Adam 優化器**像是一顆「帶有慣性的保齡球」。如果它之前一直沿著某個方向在下坡（動量 Momentum），即使突然遇到一個平地或小坑洞，它也會因為物理性慣性直接衝過去，不會被卡死在「局部最小 (Local Minima)」的假山谷裡。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要我把學習率 (Learning Rate) 設到極度小拿時間去換空間 (如 $0.000000001$)，模型最後『總有一天』會走到最完美的山谷底。**
  * *真相*：極小的學習率不僅會慢到下個世紀，而且會極度容易陷入一個很淺的「局部平原」爬不出來。這也是為什麼你的專案裡要用 `Learning Rate Scheduler (學習率衰減)`：一開始步伐邁大點 (1e-3) 大步跨越崇山峻嶺，到了後期再碎步 (1e-5) 慢慢精確走向谷底。

---

## 3. Code & Engineering (程式碼實作與工程解密)

這段程式碼將手把手展示那三行被稱為被視為 PyTorch "咒語" 的核心更新機制是如何跟模型參數交互的。

```python
import torch
import torch.nn as nn
from torch.optim import Adam

def demonstrate_backpropagation() -> None:
    """
    透過一個超微型的單神經元網路，
    展示 PyTorch 中的自動微分 (Autograd) 與 Adam 參數更新流程。
    """
    # 確保每次跑結果都一樣 (Fix Seed)
    torch.manual_seed(42)
    
    # 1. 建立一個只有單一權重 (Weights) 的神經網路層
    # Linear(1, 1) 代表: y = w * x + b
    model = nn.Linear(1, 1)
    
    # 印出大腦初始狀態 (這時它是隨機初始化的)
    print(f"--- 🧠 初始狀態的大腦權重 ---")
    print(f"Weight: {model.weight.data.item():.4f}, Bias: {model.bias.data.item():.4f}")
    
    # 2. 定義優化器 Optimizer (選擇 Adam)
    # 參數 model.parameters() 告訴優化器：「請控制這顆大腦裡的所有神經權重，並負責微調它們」
    # lr=0.1 代表它每次修正的步伐大小
    optimizer = Adam(model.parameters(), lr=0.1)
    
    # 準備一個訓練資料：輸入(x)是 2.0，我們希望標準答案(y_true)是 10.0
    x = torch.tensor([2.0])
    y_true = torch.tensor([10.0])
    
    # 3. 神聖的三聯擊 (Training Step - 這一回合只做一次)
    print("\n--- ⚡ 執行前向與反向傳播 (Forward & Backward Pass) ---")
    
    # 前向傳播 (猜測)
    y_pred = model(x)
    print(f"模型根據初始大腦猜出的答案: y_pred = {y_pred.item():.4f} (離 10.0 差遠了!)")
    
    # 計算神經網路有多蠢 (Loss)
    loss = (y_pred - y_true)**2
    print(f"計算出損失/誤差 (Loss) = {loss.item():.4f}")
    
    # [咒語一] 將這顆大腦之前的「責罵紀錄 (Gradients)」通通清空，避免將上一次的錯怪到這一次來
    optimizer.zero_grad()
    
    # [咒語二] 老闆開始往下發號施令：啟動微積分煉鎖律，一路往回算出每一個權重要負多少責任
    loss.backward()
    
    # 你可以真的去偷看 PyTorch 算出來的斜率 (Gradient)
    print(f"算出 Weight 該負的責任(斜率): {model.weight.grad.item():.4f}")
    
    # [咒語三] 部門大反省：讓 Adam 優化器根據算好的梯度去「真正修改」那顆大腦的權重！
    optimizer.step()
    
    # 我們來看看學習完一次之後，大腦變成什麼樣子了？
    print("\n--- 🧠 學習(更新)了一次之後的大腦權重 ---")
    print(f"新 Weight: {model.weight.data.item():.4f}, 新 Bias: {model.bias.data.item():.4f}")
    
    # 如果我們再拿一樣的問題去考它，會發生什麼事？
    with torch.no_grad(): # .no_grad() 是一個省記憶體的開關，代表「現在是在考試，不是在訓練，不准記下計算圖」
        new_pred = model(x)
        print(f"模型重新挑戰剛才的問題，新的預測: y_pred = {new_pred.item():.4f} (看！它往標準答案 10.0 靠近了！)")

if __name__ == "__main__":
    demonstrate_backpropagation()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **不要忘記 `optimizer.zero_grad()`：** 在 PyTorch 中，如果你忘了寫這行，梯度會「不斷被累加 (Accumulate)」。到了第三個迴圈，你的梯度會變成 $grad_1 + grad_2 + grad_3$，模型會如同吃了興奮劑一樣狂奔，導致 Loss 瞬間發散變成 `NaN`，所有權重炸裂到太陽系之外！這是一個隱蔽且最讓新手崩潰的低級失誤。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
在前述的程式碼中，為什麼在推論（評估）階段 `model(x)` 時，我們需要用一個縮排區塊包在 `with torch.no_grad():` 的下面？
如果你在跑測試集 (Test Set) 忘記加上這一行，會對這張 NVIDIA 高階顯卡的 **VRAM (顯示記憶體)** 造成什麼災難性的毀滅現象？請說明原因。

### Extreme Edge-Case (魔王挑戰)
圖神經網路中非常容易發生 **「梯度爆炸 (Gradient Explosion)」**（也就是 $\frac{\partial \mathcal{L}}{\partial w_i}$ 的值突破天際，變成一千萬甚至幾億）。
這是因為圖網路做深層 Message Passing 時，那些相乘矩陣的特徵值大於 1，透過連鎖律連乘 5 層後會指數級爆炸。
請查閱官方文件探討 PyTorch 對付梯度爆炸的最強止血鉗：`torch.nn.utils.clip_grad_norm_`。請問它是透過修改梯度的什麼數學性質來強制壓制它的破壞力，同時又「不改變原先下山的方向」的？