# 第十七章：如何駕馭社交網路（NetworkX 實戰與度中心性）

## 1. The Intuition (引言與核心靈魂)

上一堂課我們談完圖論中冷冰冰的數學定義 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 以及鄰接矩陣 $\mathbf{A}$。
現在，如果給你一個擁有 3 億用戶的 Amazon 商品購買清單資料庫，你要怎麼用 Python 把這些密密麻麻的 0 和 1 「抓在手裡」？
這時就需要請出圖論界的 Pandas——**NetworkX**。

NetworkX 是一個用來**創建、操作以及研究複雜網路結構與動態的 Python 核心套件**。
這堂課我們將學習如何計算一個節點有幾個朋友，這在圖論中叫做 **Degree (度數)**。
如果你在社交網路裡認識了 10,000 個人，你的 Degree 就是 10,000，這直接說明了你在這個網路中可能是一個「超級網紅」或是「極端重要的推薦來源」。這種影響力的測量方法被稱為 **度中心性 (Degree Centrality)**。

### Learning Objectives (學習目標)
1. **支配 NetworkX 基本功**：學會加入節點、畫上邊、以及印出 Graph 的資訊。
2. **掌握 Degree 的本質**：理解每個節點的連接數量（Degree）為什麼是圖論特徵工程的命脈。
3. **洞察網路結構**：了解為什麼有的人有 100 萬個粉絲，有的人只有 3 個（Power-Law Distribution）。

---

## 2. Deep Dive (核心概念與深度解析)

### NetworkX: The Standard Library of Graphs
在 GNN 深度學習盛行前，幾乎所有的圖論演算法（最短路徑 Dijkstra's, PageRank, 社群偵測）都是用純 CPU 在跑，而 Python 世界的主流工具就是 NetworkX。即使現在我們用 PyTorch Geometric (PyG)，你也經常需要把圖轉換成 NetworkX 來進行視覺化（Draw Node/Edge）或圖表分析。

### Degree (度) 
對於無向圖 (Undirected Graph) 中的任何一個節點 $v$，它的度表示為 $deg(v)$，代表與其直接相連的邊數。
如果我們觀察先前的鄰接矩陣 $\mathbf{A}_{N \times N}$：
$$ deg(v_i) = \sum_{j=1}^{N} \mathbf{A}_{i, j} $$
（把矩陣第 $i$ 列的數字全部加起來就是這個節點的 Degree）

對於有向圖 (Directed Graph) 來說，Degree 會被一分為二：
*   **In-Degree (入度)**：有幾條連線「指進來」。在 IG 上，這叫作 Followers（粉絲數）。
*   **Out-Degree (出度)**：有幾條連線「指出去」。在 IG 上，這叫作 Followings（你追蹤的人數）。

### 分佈法則 (Degree Distribution)
在真實世界（Amazon 商品網、小紅書、Twitter）中的圖，通常呈現 **「冪律分佈 (Power-law Distribution)」或是「無標度網路 (Scale-Free Network)」**。
這意味著：有極少數的節點（Hub Nodes，比如 Justin Bieber、Amazon的 iPhone 充電線）擁抱著極度誇張的天文數字連線，而絕大多數的節點（邊緣人、冷門書籍）只有 1 到 2 個連線。這對神經網路的訓練是一個超級天坑（Hub Nodes 會把鄰居的特徵全部吃光），我們將在之後探討。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：神經網路只看特徵 (Features)，不需要管 Degree，它可以自己學會。**
  * *真相*：這是大錯特錯。在 GCN (圖卷積) 這樣的模型中，每一次更新自己都是要把「鄰居傳過來的情報」加在一起。如果一個節點擁有 10,000 個鄰居，相加之後它的數值會變得超巨大而「爆炸 (Explode)」；如果它只有 1 個鄰居，數值會太小。如果你不懂 Degree，你根本看不懂 GCN 公式中那個神經病的 $1/\sqrt{deg(i) \times deg(j)}$ (Normalization) 到底在除什麼鬼東西。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們現在要用 NetworkX 建立一個圖，並且實際算算看誰是這個網路的老大。

```python
import networkx as nx

def analyze_friend_network() -> None:
    """
    使用 NetworkX 建立一份社交網路，並尋找其中的網紅 (High Degree Hub)。
    """
    
    # 1. 召喚網路物件
    # nx.Graph() 是無向圖; nx.DiGraph() 則是有向圖
    G = nx.Graph()
    
    # 2. 我們一口氣加入很多條邊 (加邊的時候，NetworkX 會自動創建不存在的節點)
    print("🕸️ 建立社交網路連線中...")
    G.add_edges_from([
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Alice", "David"),
        ("Bob", "Charlie"),
        ("Eve", "Alice"),  # Alice 又多認識了一個人!
    ])
    
    # 3. 基礎窺探 (Basic Info)
    print("\n--- 網路掃描報告 ---")
    print(f"總共登錄的人數 (Nodes): {G.number_of_nodes()}")
    print(f"總共建立的關係 (Edges): {G.number_of_edges()}")
    
    # 4. 度分析 (Degree Analysis) 尋找誰是最具有宰制力的人
    print("\n--- 影響力排行榜 (Degree) ---")
    
    # G.degree 會回傳類似 dict 的物件: {'Alice': 4, 'Bob': 2, ...}
    for person, num_friends in G.degree:
        print(f"📊 {person} 總共有 {num_friends} 個朋友。")
        
    # Python 魔法：找出字典中 Value 最大的那個 Key 
    # max() 參數傳入 iter，並且自訂 key function 就是取 G.degree[node] 的數字大小
    influencer = max(dict(G.degree).items(), key=lambda item: item[1])
    print(f"\n👑 [系統判定] 本社群最危險的中心人物 (Hub) 是: {influencer[0]} (擁抱 {influencer[1]} 條連線)")

if __name__ == "__main__":
    analyze_friend_network()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **NetworkX 太慢了：** 非常重要。NetworkX 完全是用純 Python 撰寫的，而且設計上極端優雅但效能很差。當你的節點超過數十萬個（例如百萬規模的 Amazon 圖），跑一個 NetworkX 的演算法可能會耗費整整一天。在真實的工程中，我們遇到超大圖的時候，會改用 C++ 寫成的套件（例如 **iGraph**、**cuGraph (GPU加速)**）或者用 PyTorch 的稀疏矩陣 (Sparse Tensor) 來手寫運算。 

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
如果一張網路中有 100 個節點，而且這是一張 **「全連接圖 (Fully Connected Graph 或稱 Complete Graph $K_{100}$ )」**，也就是每個人都認識除了自己以外的所有人。
請問：
1. 這個圖中隨便挑一個節點的 Degree 會是多少？
2. 整個網路中的邊數 (Edges) 總共有幾條？ (提示：握手問題，不要重複計算)。

### Extreme Edge-Case (魔王挑戰)
你現在是一家銀行的洗錢防制工程師。你建立了一個有向圖 `nx.DiGraph()`，節點是銀行帳戶，邊的箭頭 $A \rightarrow B$ 代表 A 匯款給 B。
如果你看到一個神奇的帳戶 X：
它的 In-Degree 等於 10,000 (無數筆小額匯款流入)；
它的 Out-Degree 等於 1 (一筆超巨大額度直接轉到一個海外不知名帳戶)。
請你從剛剛學過的「出度」與「入度」結構特性來解釋，這是不是洗錢防制系統（AML）中典型可以用圖論直接標記的犯罪樣態（Money Mule / Smurfing 結構）？試著想看看。