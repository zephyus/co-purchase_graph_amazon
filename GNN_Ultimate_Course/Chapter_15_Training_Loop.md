# 第十五章：手工刻一個最簡單的 PyTorch 訓練迴圈

## 1. The Intuition (引言與核心靈魂)

如果我們把前面四章學到的東西當作汽車的零件：
我們已經有了引擎（模型神經網路, Neural Networks）、有了解讀儀表板油量的感測器（損失函數, Loss Function）、也有了能微調方向盤的智駕系統（優化器, Optimizer Adam），以及承載著我們目的地地圖的記憶體空間（張量, Tensors）。

現在，我們終於要把鑰匙轉動，發動這輛車。這個發動的過程就是 **「訓練迴圈 (Training Loop)」**。
在深度學習中，訓練迴圈是一個極其儀式化 (Ritualistic) 的 5 步流程。這 5 個步驟在全世界所有的 PyTorch 研究室裡，無論是預測股票、辨識貓狗，還是像你的 Amazon GNN 專案做節點推薦，寫法本質上都**完全一樣**。學會這個迴圈，你就等於拿到了驅動當代 AI 的萬能鑰匙。

### Learning Objectives (學習目標)
1. **內化神聖 5 步驟**：Forward、Loss、Zero-grad、Backward、Step 倒背如流。
2. **Epoch vs Batch**：清楚區分「讀完一本書」與「讀完一頁」的差異。
3. **區分模式 (Mode Toggle)**：掌握 `model.train()` 與 `model.eval()` 對網路深層機制的生死影響。

---

## 2. Deep Dive (核心概念與深度解析)

### Epoch 與 Batch 的時間度量學
在機器學習的時間線中，有兩個我們用來標記壽命的字眼：
* **Epoch (紀元)**：神經網路把「整套訓練資料集 (整個題庫)」完完整整地看過一次，叫做一個 Epoch。
* **Batch (批次)**：如果題庫有十萬題，一次全塞進 GPU 會直接 OOM (Out Of Memory) 爆炸。所以我們會把它切成每次 512 題為一疊送進去。這疊 512 題就是一個 Batch。

當一個模型經過了 `Epoch = 200`，表示它已經把同一個題庫反覆讀了兩百遍。如果到了這個時候 Training Loss 還是很高，這叫做**欠擬合 (Underfitting)**（代表這模型的大腦容量太小或太笨）；如果 Training Loss 變成了 0.001 但 Test Loss 卻飆高，這就是**過擬合 (Overfitting)**（它變成了只會死背的書呆子）。

### Dropout 與 Train/Eval 模式切換
這是一個在 GNN 中極為致命的概念陷阱。
為了對抗上面提到的過擬合，研究人員發明了 **Dropout (隨機失活)** 技術，在「訓練期間」，我們會隨機把大腦中 20% 的神經元「打暈（強制設為 0）」，逼迫剩下的 80% 神經元自立自強學習；
但是，當我們要上考場時（Validation 或是 Test 預測階段），我們必須要把所有神經元喚醒，**100% 滿血上陣**。

這也就是為什麼在 PyTorch 中，我們必須反覆大喊這兩句咒語：
*  `model.train()` $\xrightarrow{}$ 告訴它「現在在訓練，給我把 Dropout 開啟！」
*  `model.eval()`  $\xrightarrow{}$ 告訴它「準備期末考，關閉 Dropout，用真實力上場！」
如果你在產生期末考結果時忘了寫 `model.eval()`，你的模型分數每次跑出來都會莫名其妙上下劇烈浮動（因為有 20% 的腦細胞正在睡覺）。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要我寫了 `with torch.no_grad():`，模型就會自動從訓練模式跳成測試模式。**
  * *真相*：這是兩個完全不同維度的東西！`no_grad()` 只負責「關閉梯度記憶流」來節省 VRAM 記憶體；而 `model.eval()` 負責「改變網路內部層的物理行為 (如 Dropout, BatchNorm)」。在你的驗證測試函數裡，這兩個都必須同時存在，缺一不可。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們現在要將之前的積木組合起來，寫下一個具備**訓練、驗證、防禦 OOM** 的史詩級微型訓練迴圈範本。這個框架，基本上就是你 Amazon GNN 裡 `train.py` 的靈魂縮影。

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# 假設這是一個隨機虛構的極微型資料集與模型 (用來展示 Loop，非真實 GNN)
def build_dummy_network(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(p=0.2), # 加入防止死背的 Dropout (20% 斷線)
        nn.Linear(64, 1)
    )

def run_golden_training_loop() -> None:
    """執行一個完美無瑕的神聖 5 步訓練迴圈範本。"""
    
    # 0. 環境綁定與裝備初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_dummy_network(input_dim=10).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 模擬 500 筆訓練資料 (Train Set) 與 100 筆測試資料 (Val Set)
    X_train = torch.randn(500, 10).to(device)
    y_train = torch.randint(0, 2, (500, 1)).float().to(device) # 二元分類: 0 或 1
    
    X_val = torch.randn(100, 10).to(device)
    y_val = torch.randint(0, 2, (100, 1)).float().to(device)
    
    # 開始紀元迴圈 (Iterate over Epochs)
    epochs = 100
    print(f"--- 🚀 準備啟動引擎。執行 {epochs} 個 Epoch ---")
    
    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------
        # 🚀 [訓練階段 Train Phase]
        # ----------------------------------------------------
        model.train() # 【關鍵】開啟 Dropout 等訓練專用陷阱
        
        # [步驟 1/5] 前向傳播 (Forward Pass) -> 網路給出瞎猜分數
        logits = model(X_train)
        
        # [步驟 2/5] 算帳 (Calculate Loss) -> 誤差尺衡量偏差
        loss = criterion(logits, y_train)
        
        # [步驟 3/5] 清空舊恨 (Zero Gradients) -> 不要把上一次的錯怪到這次來
        optimizer.zero_grad()
        
        # [步驟 4/5] 追本溯源 (Backward Pass) -> 激發鏈鎖律，找出誰該負責
        loss.backward()
        
        # [步驟 5/5] 行動修正 (Optimizer Step) -> 微調大腦權重
        optimizer.step()
        
        # ----------------------------------------------------
        # 🛡️ [驗證階段 Validation Phase]
        # ----------------------------------------------------
        # 每過 20 個 Epoch，讓它考一次模擬考，看看有沒有過擬合
        if epoch % 20 == 0:
            model.eval() # 【關鍵】關閉 Dropout，叫醒所有神經元！100% 滿血！
            
            with torch.no_grad(): # 【關鍵】告訴 PyTorch 不要浪費記憶體建構微積分樹
                # 給它我們準備好的「沒見過的」期末考題
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val)
            
            print(f"[Epoch {epoch:3d}] Train Loss: {loss.item():.4f} | Validation Loss: {val_loss.item():.4f}")

if __name__ == "__main__":
    run_golden_training_loop()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **Early Stopping (早停法)：** 在實際的 Amazon 專案中，我們不會像個笨蛋一樣設定了 200 個 Epoch 就乖乖等它跑完。在 `Validation Phase` 中，如果我們發現程式已經連續 20 個 Epoch 的 Validation Loss 不降反升（儘管 Train Loss 還在下降），表示模型已經開始「走火入魔瘋狂背題庫」了！這時我們必須觸發名為 Early Stopping 的機制，呼叫 `break` 踢破迴圈提早結束訓練，並把歷史上跑出最好 Validation 分數的那一個模型檔案 (`best_model.pt`) 存起來，而不是留存最後一天那個過擬合的爛大腦。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
在前向傳播 `logits = model(X_train)` 和計算誤差 `loss = criterion(...)` 中，其實並沒有更新到任何關於 `model` 權重裡的數字。
請用你自己的話，總結第 3 到第 5 步 (`zero_grad()`, `backward()`, `step()`) 分別在電腦中發動了什麼樣的純數學底層操作才完成了這一次的大腦微調？

### Extreme Edge-Case (魔王挑戰)
如果有一天，你的老闆要求你要在一張記憶體只有 8GB 的破顯卡上面，訓練一個光是模型權重就吃掉 7GB 的巨大模型（例如 BERT）。
這時即使你將 Batch Size 調低到了非常小的數值（例如 2），它依然在執行第二步迴圈時彈出了 OOM (Out Of Memory) 記憶體爆炸的死亡紅字。
請去查閱深度學習中的高級救命技巧 **「梯度累積 (Gradient Accumulation)」**。它如何利用「打破神聖 5 步原有的絕對綁定關係（例如讓 `backward()` 跑三次才執行一次 `step()` 與 `zero_grad()`）」，從而在極小的記憶體硬體上模擬出超大 Batch Size 的穩定訓練效果？這背後的運算代價又是什麼？