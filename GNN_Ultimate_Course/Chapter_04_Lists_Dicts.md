# 第四章：整理百寶袋（清單 List 與 字典 Dictionary）

## 1. The Intuition (引言與核心靈魂)

如果我們只有單一變數（如 `a = 5`），那麼要在電腦裡模擬現實世界的複雜結構幾乎是不可能的。想像你要記錄一整條街道上 100 戶人家的地址，你難道要宣告 `house1_地址`, `house2_地址` 一直到 `house100_地址`？
這時你需要的是 **資料容器 (Data Structures)**。

在 Python 裡，**清單 (List)** 就像是一列有編號的火車車廂，它強調的是「順序 (Order)」與「位置 (Index)」；而 **字典 (Dictionary)** 則像是一本真正的牛津字典或者查號台通訊錄，它不關心順序，它關心的是「關鍵字 (Key)」如何快速對應到「解釋/數值 (Value)」。

當你未來在做 Amazon 的圖論模型時，你需要記錄「哪些節點是哪些節點的鄰居」：如果你用錯了容器（例如在一個龐大的 List 裡面用 `in` 去搜尋一個節點是否存在），那你的神經網路光是「找資料」就要花費好幾個小時；如果你懂得用 Dictionary 或 Set 這種 Hash 結構，這個搜尋動作只需要 0.001 秒。

### Learning Objectives (學習目標)
1. **掌握 List/Tuple**：學習 0-based 索引 (0-based indexing)、切割 (Slicing) 以及原地修改 (In-place modification)。
2. **精通用查表法 (Dictionary)**：理解 Key-Value 配對模型，並掌握 `.get()`, `.keys()`, `.items()` 的用法。
3. **理解時間複雜度 (Big O) 的差異**：在不同資料結構中搜尋元素的速度差距。

---

## 2. Deep Dive (核心概念與深度解析)

資料結構決定了演算法的極限。這裡我們要從計算機科學底層談起：

### List (陣列/列表)
在 Python 內部，`list` 實際上是一個動態陣列 (Dynamic Array)，它在記憶體中存放的並非物件本身，而是一連串連續的「記憶體指標 (Pointers)」。
當你執行 `my_list[3]` 時，電腦只要計算 `起始位址 + 3 * 指標大小`，就能以 $O(1)$ 常數時間瞬間跳到那個位置並提取資料。
然而，如果你要問這輛火車裡有沒有裝載「神經網路」這個詞 (`"神經網路" in my_list`)？電腦只能從第一節車廂開始一節節地檢查，最差情況下必須走完 $N$ 節車廂，時間複雜度淪為 $O(N)$。

### Dictionary (字典 / Hash Map)
字典 (`dict`) 的設計則是為了解決「搜尋太慢」的問題。它依賴 **雜湊函數 (Hash Function)**。
設 $\mathcal{K}$ 為鍵的集合，$\mathcal{V}$ 為值的集合。
我們定義映射 $H: \mathcal{K} \rightarrow \mathbb{N}$。
當你想把 `Key = "node_104"` 對應到 `Value = [1, 5, 9]` 時，我們把 `"node_104"` 丟進雜湊函數中，算出一個整數，直接算出該放到記憶體的第幾個抽屜去。
下次你想查詢 `"node_104"` 時，只要再算一次 $H(\text{"node_104"})$ 就能立刻知道去哪個抽屜拿資料。這使得 `key in my_dict` 的時間複雜度戲劇性地降至 $O(1)$！

### 🚨 Common Misconceptions (常見迷思)
* **迷思：List 和 Dictionary 裡面只能放整數或字串。**
  * *真相*：上面可以放「任何東西」。你可以放一個 List 進 Dictionary 裡成為 Value（用來儲存圖論中的鄰接表 Adjacency List），你甚至可以把「函式」當作物件存進 List 裡去依次執行。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們來建立一個極簡單版的「圖網路結構 (Graph)」來示範如何使用 Dictionary 來儲存鄰居節點。

```python
from typing import Dict, List, Set

def build_adjacency_list() -> None:
    """
    使用 Dictionary 與 List 來建立一個無向圖 (Undirected Graph) 的鄰接表。
    這是 GNN 在進入 PyTorch Geometric 之前最原始的表示方式。
    """
    
    # 建立圖的字典: Key 是節點 ID (int), Value 是該節點的鄰居清單 (List[int])
    graph: Dict[int, List[int]] = {
        0: [1, 2, 3],
        1: [0, 4],
        2: [0],
        3: [0, 4, 5],
        4: [1, 3],
        5: [3]
    }
    
    # 1. 字典的遍歷 (Iterating over Dictionary)
    print("--- 圖的鄰居節點 ---")
    for node_id, neighbors in graph.items():
        print(f"節點 {node_id} 的鄰居有: {neighbors} (Degree = {len(neighbors)})")
    
    # 2. 安全的取值方法: dict.get()
    # 如果我們直接呼叫 graph[99]，因為 99 不存在，程式會崩潰 (KeyError)
    # 但使用 .get()，我們可以給予它防呆的預設值
    target_node = 99
    safe_neighbors = graph.get(target_node, [])
    print(f"\n嘗試取得節點 {target_node} 的鄰居: {safe_neighbors}")
    
    # 3. List Comprehension (清單推導式) - Python 最強大的裝備
    # 我們想算出所有節點各自有多少鄰居 (Degree)
    degrees = [len(n) for n in graph.values()]
    print(f"\n所有節點的 Degree 分布: {degrees}")
    print(f"圖中總節點數: {len(graph)}")

if __name__ == "__main__":
    build_adjacency_list()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **為何 Dictionary 的 Key 不能是 List? (Unhashable Types)：** 
  因為雜湊表的安全性建立在「Key 不能在日後偷偷被修改」。如果你用一個 List 當作 Key，而在某個地方往 List 裡塞了一個新元素，它的 Hash 值就會改變，整個字典就會崩壞。所以 Python 嚴格規定：**只有不可變 (Immutable) 的物件（如 int, str, tuple）才能作為 Dictionary 的 Key**。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
針對 Python 中的 List Slice (切片) 語法 `list[start:stop:step]`。
若有一個 `sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`，請不寫程式預測這兩個語法的輸出：
1. `sequence[::-1]`
2. `sequence[2:8:2]`
這在後續 PyTorch 張量（Tensor）中分割訓練集與測試集時是天天在用的基礎語法。

### Extreme Edge-Case (魔王挑戰)
在 NLP 或是大數據圖論中，節點的數量可能超過百萬個，有些節點甚至可能出現「重複連線」（例如 Amazon 裡面使用者不小心點擊了商品兩次）。
請查閱 Python 中的 `Set` (集合) 資料型態。若我們想儲存「不允許重複」的邊緣 (Edges) 集合，為什麼會建議把 `List` 改成用 `Set` 來管理鄰居？若將上述實作碼從 `graph: Dict[int, List[int]]` 變成 `graph: Dict[int, Set[int]]`，這對於計算節點到節點是否有連線的檢查 `$O(?)$` 會產生怎樣的效能劇變？