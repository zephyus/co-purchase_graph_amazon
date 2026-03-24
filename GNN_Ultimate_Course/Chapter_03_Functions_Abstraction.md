# 第三章：重製武器的熔爐（函式 Function）

## 1. The Intuition (引言與核心靈魂)

如果每次你想要吃蛋糕，你都必須親自去買麵粉、買糖、打蛋、生火、烘烤，那你幾乎什麼事都不用做了。更好的做法是，找一位叫做「烘焙機」的機器，你只要把麵粉和糖（**輸入參數 (Inputs)**）丟進去，按下按鈕（**呼叫呼叫 (Call)**），過一會兒它就會吐出一個完美的蛋糕（**輸出回傳值 (Returns)**）。

在寫程式時，如果你發現自己將同一段判斷邏輯複製貼上超過兩次，那這段程式碼就應該被封裝成一個「函式 (Function)」。函式就像是一個重製武器的熔爐，它不僅能大幅減少程式碼的冗餘，更重要的是，它隱藏了實作的複雜度 (Abstraction)；當我們在設計複雜的神經網路時，我們只關心「這個函式負責做前向傳播 (Forward Pass)」，而不需要在主程式中看到所有微積分的細節。

### Learning Objectives (學習目標)
1. **理解模組化思維 (Modularity)**：學會定義 (def) 與呼叫函式，隔離作用域 (Scope)。
2. **掌握參數系統**：熟悉位置參數 (Positional)、關鍵字參數 (Keyword)、預設參數 (Default) 與不定長度參數 (`*args`, `**kwargs`)。
3. **理解 Return 的本質**：區分「印出 (print)」與「回傳 (return)」在資料流操作上的根本差異。

---

## 2. Deep Dive (核心概念與深度解析)

從數學分析的角度來看，一個 Python 函式是對數學函數 $f(x)$ 概念的工程擴展。
給定定義域集 (Domain) $X$ 與對應域集 (Codomain) $Y$。若函式定義為 $f: X \rightarrow Y$，則對於每一個輸入 $x \in X$，都會映射到唯一的一個輸出 $y \in Y$。

但在程式設計中有所不同，Python 函式除了進行映射，還可能產生 **副作用 (Side Effects)**。副作用指的是函式在執行過程中，修改了外部的狀態（例如修改了傳入的可變清單，或是在螢幕上印出了文字）。
在數學上純粹的函數（如 Haskell 語言中的純函數）保證：
$$ f(x) = y $$
每次輸入相同的 $x$ 永遠得到相等的 $y$，且不影響外界。
但在 Python 與機器學習領域，我們經常依賴副作用（例如呼叫 `model.train()` 會改變神經網路內部的權重狀態矩陣 $W$ 及 $b$）。理解哪些函式是純粹的，哪些帶有副作用，是避免除錯地獄的關鍵。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：如果你希望程式外的人看到結果，就應該在函式裡面寫 `print()`。**
  * *真相*：`print()` 只是將文字輸出到使用者的終端機螢幕，這些資料一旦印出，程式的其它部分就再也無法使用它。如果你希望另一段程式碼能接收這些計算結果，你**必須**使用 `return`。沒有寫 `return` 的函式預設會回傳一個虛無的 `None` 物件。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們來寫一個真正具備工業水準的 Python 函式。這次我們設計一個用來計算「神經網路學習率衰減 (Learning Rate Decay)」的工具函式。它示範了 Type hints、Docstrings（API文件），以及如何優確處理除外邊界。

```python
from typing import Optional

def calculate_decayed_lr(
    initial_lr: float, 
    epoch: int, 
    decay_rate: float = 0.9, 
    min_lr: float = 1e-6
) -> float:
    """
    根據指數衰減公式計算特定 epoch 時的學習率。
    
    公式: lr_current = initial_lr * (decay_rate ** epoch)
    如果算出的學習率低於 min_lr，則自動截斷於 min_lr 以防止梯度停止更新。
    
    Args:
        initial_lr: 初始學習率
        epoch: 當前的訓練週期 (必須 >= 0)
        decay_rate: 每過一個 epoch 的衰減乘數 (通常介於 0 與 1 之間)
        min_lr: 學習率的下界 (Lower bound)
        
    Returns:
        float: 經過衰減後的當前學習率
        
    Raises:
        ValueError: 當 epoch 為負數時拋出例外
    """
    
    # 1. 參數合法性防禦 (Defensive Programming)
    # 在執行核心數學之前，先擋掉非法的輸入，防止程式在幾小時後無預警崩潰
    if epoch < 0:
        raise ValueError(f"Epoch 不能是負數！收到: {epoch}")
        
    # 2. 核心數學運算
    current_lr = initial_lr * (decay_rate ** epoch)
    
    # 3. 邊界處理 (Clipping)
    # 其實也可以使用 PyTorch/Numpy 內建的 clip 函式，但在純 Python 中可以這樣寫
    if current_lr < min_lr:
        return min_lr
        
    return current_lr

if __name__ == "__main__":
    init_lr = 0.1
    # 模擬訓練經過 10 個與 50 個 Epoch 後的狀態
    print(f"Epoch 10 的 Learning Rate: {calculate_decayed_lr(init_lr, epoch=10):.6f}")
    
    # 透過明確指定參數名稱 (Keyword Arguments)，代碼可讀性大幅提升
    print(f"Epoch 50 的 Learning Rate: {calculate_decayed_lr(initial_lr=init_lr, epoch=50, decay_rate=0.8):.6f}")
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **危險的預設可變參數 (Mutable Default Arguments)：**
  如果你寫 `def append_node(node_id, my_list=[]): my_list.append(node_id)`，這會引發極度可怕的 Bug。因為在 Python 中，預設參數是在「函式被定義」的當下就被建立並保存在記憶體中了。這意味著所有對這個函式的呼叫，如果不傳入自訂清單，都會「共用」同一個 `my_list` 記憶體。**工程解法**：永遠用 `def append_node(node_id, my_list=None): if my_list is None: my_list = []`。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
考慮以下兩個函式的差異：
```python
def add_nodes_pure(nodes_list, new_node):
    return nodes_list + [new_node]

def add_nodes_side_effect(nodes_list, new_node):
    nodes_list.append(new_node)
    return nodes_list
```
如果你將同一個名為 `my_graph_nodes` 的清單傳入這兩個函式，請問外部原本的清單在兩者執行後發生了什麼不同的變化？為什麼在訓練神經網路時我們更傾向於控制「副作用」？

### Extreme Edge-Case (魔王挑戰)
在機器學習的複雜配置中，我們經常看到這種宣告方式：
`def build_model(in_channels, out_channels, *args, **kwargs):`
請詳盡解釋 `*args` (星號參數) 與 `**kwargs` (雙星號關鍵字參數) 分別會把多餘傳入的變數轉換成 Python 中的什麼資料結構（Tuple 還是 Dictionary）？並嘗試自己寫一個能接受無窮無盡數量的 `**kwargs` 然後把它們的鍵與值印出來的函數。