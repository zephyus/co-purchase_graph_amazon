# 第二章：程式的十字路口與迴圈（if/else 與 for loop）

## 1. The Intuition (引言與核心靈魂)

想像你正在玩一款複雜的角色扮演遊戲。你在路上遇到一個守衛，他問：「你有通行證嗎？」如果你選擇「有」，他會為你開門；如果你選擇「沒有」，他會拔劍攻擊。這就是程式中的「條件判斷 (if/else)」，它賦予了程式靈魂，使其具備**決策能力**。

過了一會兒，你來到一個果園，你要採收 100 顆蘋果。你不可能對你的身體下達 100 次「採一顆蘋果」的獨立指令，而是會大腦下達一個指令：「重複採蘋果這個動作，直到滿 100 顆為止」。這也就是程式中的「迴圈 (Loop)」，它賦予了程式**自動化與擴展能力**。沒有條件判斷與迴圈，電腦就只是一個巨大的算盤；有了它們，電腦才能成為智慧的代理人。

### Learning Objectives (學習目標)
1. **精準掌握邏輯控制流**：學會使用 `if`, `elif`, `else` 建構複雜決策樹。
2. **自動化重複任務**：深究 `for` 與 `while` 迴圈的使用時機。
3. **理解迭代器本質**：理解 Python 如何利用 `range()` 與可迭代物件 (Iterables) 達到極高效率的記憶體應用。

---

## 2. Deep Dive (核心概念與深度解析)

從運算理論 (Computability Theory) 的角度來看，任何具備圖靈完備 (Turing Complete) 系統的語言，都必須具備基本的條件分支與迴圈能力。

### 條件判斷的邏輯代數
條件判斷 `if condition:` 依賴的是布林代數 (Boolean Algebra)。設條件為命題變數 $p, q$，我們可以使用邏輯算子 $\land$ (AND), $\lor$ (OR), $\neg$ (NOT) 組合複雜條件。
在 Python 執行時，直譯器會計算條件表示式的真值 (Truth Value)，將其映射到集合 $\{ \text{True}, \text{False} \}$。

### 迴圈的時間複雜度 (Time Complexity)
當我們寫下一個迴圈時，我們同時也引入了時間複雜度的考量。
設迴圈的迭代次數為 $N$。
```python
for i in range(N):
    f(i) # 執行某操作
```
若 $f(i)$ 的執行步數為常數 $O(1)$，則整體演算法的時間複雜度即為 $O(N)$。在未來處理上千萬節點的圖論資料 (如 Amazon dataset) 時，如果在 Python 層級寫了雙層迴圈 $O(N^2)$，程式可能要跑上好幾個世紀，這就是為什麼後續我們需要學習向量化運算 (Vectorization)。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：`for i in range(1000000000)` 會在一瞬間耗盡記憶體，因為它產生了十億個數字。**
  * *真相*：在 Python 3 中，`range()` 是一個生成器 (Generator) 結構。它並不會真正在記憶體裡面建立一個含有十億個數字的列表。它只儲存了 `start`, `stop`, `step` 三個數字，然後每次迴圈走到時，才「即時運算 (Lazy Evaluation)」出下一個數字。這是一個極其重要的記憶體優化概念。

---

## 3. Code & Engineering (程式碼實作與工程解密)

這段程式碼展示了如何將條件控制與迴圈完美結合。在資料科學中，這常被用於「清理資料」或是「初步過濾邊緣集合」。

```python
from typing import List, Tuple

def filter_high_degree_nodes(nodes_degrees: List[Tuple[int, int]], threshold: int = 5) -> List[int]:
    """
    過濾並回傳連線數 (degree) 大於或等於門檻值的節點 ID。
    
    Args:
        nodes_degrees: 每個元素為 (節點ID, 連線數) 的列表
        threshold: 判定為高連通性節點的門檻值 (預設為 5)
        
    Returns:
        List[int]: 符合條件的節點 ID 列表
    """
    vip_nodes: List[int] = []
    
    # 這裡的 for 結合了 tuple unpacking，是非常 Pythonic 的寫法
    for node_id, degree in nodes_degrees:
        
        # 條件分支：如果度數過低，使用 continue 提早跳過當次迭代，減少巢狀縮排 (Early Return 原則)
        if degree < 0:
            print(f"警告：節點 {node_id} 擁有異常的負數度數 ({degree})，略過處理。")
            continue
            
        if degree >= threshold:
            vip_nodes.append(node_id)
        else:
            # 雖然不需要做任何事，但寫 pass 有助於保留未來擴展的彈性
            pass
            
    return vip_nodes

if __name__ == "__main__":
    # 模擬從圖的 dataset (例如 edges.csv 統計後) 獲得的資料
    sample_data: List[Tuple[int, int]] = [
        (0, 10), (1, 2), (2, -1), (3, 8), (4, 4), (5, 105)
    ]
    
    result = filter_high_degree_nodes(sample_data, threshold=5)
    print(f"篩選後的高影響力節點: {result}")
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **While 迴圈的無限地獄 (Infinite Loops)：** 在機器學習訓練中，如果你寫了一個 `while not converged:`，一旦演算法不穩定 (如 Loss 發散到 NaN)，條件永遠不會滿足，程式就會卡死。防禦性程式設計 (Defensive Programming) 要求你一定要設定一個 `max_iterations` 來作為安全跳出機制：`while not converged and step < MAX_STEP:`。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
判斷以下 Python 程式的輸出，並說明原因：
```python
x = [1, 2, 3]
for item in x:
    if item == 2:
        x.remove(item)
print(x)
```
*提示：思考一下當你在用迴圈遍歷一個清單時，同時又在修改這個清單的長度，會發生什麼指標錯位的災難？*

### Extreme Edge-Case (魔王挑戰)
Python 中有一種特殊的語法叫做 `for...else`。請研讀 Python 官方文件，並用 `for...else` 語法實作一個「尋找質數 (Prime Number)」的演算法。它的 `else` 區塊在什麼極端情況下**不會**被執行？請解釋為什麼這個設計語法比起使用額外的 `is_prime = True/False` 旗標變數來得更優雅。