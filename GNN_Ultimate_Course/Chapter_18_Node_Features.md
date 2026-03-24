# 第十八章：特徵工程與節點屬性（Node Features）

## 1. The Intuition (引言與核心靈魂)

到目前為止，不管是在鄰接矩陣還是 NetworkX，我們看到的圖都只是一張空殼。
我們知道「A 連接了 B」，但是 **A 到底是誰？**
在 Facebook，A 可能是 25 歲的軟體工程師，有著高收入且住在西雅圖；在 Amazon，A 可能是一本「深度學習聖經」，標籤是「電腦科學」，定價 1,200 元。

光是知道兩個人是朋友 (Edges) 是不夠的。如果我們想讓神經網路推論出「他們兩個人是否都會對某個科技產品感興趣」，我們必須賦予他們靈魂：**節點特徵 (Node Features)**。
在接下來的 GNN 模型中，節點的特徵會沿著「圖的連線」就像是電流一樣四處傳遞、融合。我們必須將這些真實世界的屬性，轉換為神經網路唯一看得懂的語言：**向量 (Vectors) $\mathbf{x_i}$**。

### Learning Objectives (學習目標)
1. **結合屬性與圖結構**：如何在數學和程式上把高維特徵（如年齡、價格、文字 Embeddings）綁定在圖節點上。
2. **理解特徵矩陣 $\mathbf{X}$**：見證從 N 個節點到 $N \times D$ 維度特徵矩陣的偉大跨越。
3. **認識 Graph 的真正力量**：為何單純使用傳統 ML（如 XGBoost）忽略圖結構會慘敗？圖如何彌補特徵的不足。

---

## 2. Deep Dive (核心概念與深度解析)

### 特徵矩陣 (Feature Matrix $\mathbf{X}$)
在現代圖神經網路 (GNN) 理論中，一張圖的完整定義不再只是 $G = (V, E)$，而是：
$$ \mathcal{G} = (\mathbf{A}, \mathbf{X}) $$
其中：
*   **$\mathbf{A} \in \mathbb{R}^{N \times N}$ 或 $\mathbf{E}$ (Edge Index)**：代表圖的「拓撲結構 (Topology)」，也就是連線關係。
*   **$\mathbf{X} \in \mathbb{R}^{N \times D}$**：代表節點的「特徵矩陣 (Feature Matrix)」。

假設有 $N = 3$ 個 Amazon 商品，每個商品有 $D = 4$ 種特徵（例如：價格、星等、類別A、類別B），那麼特徵矩陣 $\mathbf{X}$ 可能長這樣：
$$ \mathbf{X} = \begin{bmatrix} 29.99 & 4.5 & 1.0 & 0.0 \\ 59.00 & 3.8 & 0.0 & 1.0 \\ 15.50 & 4.9 & 1.0 & 0.0 \end{bmatrix} $$
矩陣的第 $i$ 列 (Row $i$)，就是節點 $v_i$ 的專屬屬性向量 $\mathbf{x_i}$。

### 為什麼 GNN 需要 $\mathbf{A}$ 和 $\mathbf{X}$ 雙管齊下？ (結構與特徵的聯姻)
這被稱為 **「同質性假設 (Homophily Assumption)」**：物以類聚，人以群分。
如果我們只看薪水 $\mathbf{X}$，傳統機器學習（像你在第六章學的 Logistic Regression）會把所有高薪的人預測為會買奢侈品。但這是不夠的！
如果一個中產階級（他自己的 $\mathbf{x_i}$ 薪水不高），但他在圖上的鄰居（他的朋友）全都是家財萬貫的富豪（鄰居的 $\mathbf{x_j}$ 薪水極高），他在這種社交圈的耳濡目染下，極有可能也會因為「結構」的壓力而去購買奢侈品！這就是圖網路會贏傳統 ML 的根本原因。我們在把節點原本低微的特徵，利用四周的神隊友進行「特徵擴充」。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要把文字像是「深度學習書」直接存進節點的屬性裡面就可以了。**
  * *真相*：神經網路看不懂字串 (`"深度學習書"`)！這也是無數資料科學家死掉的地方。在進入 PyTorch 或 GNN 之前，所有文字、類別（如國家）都必須被轉換為浮點數向量（例如 Word2Vec, BERT embeddings 或是 One-Hot Encoding）。你的 $D$ 維度裡面絕對不允許出現英文字母。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們在 NetworkX 裡面創建一個圖，並且把特徵綁定在每個節點身上，最後抽出來變成矩陣 $\mathbf{X}$。

```python
import networkx as nx
import numpy as np

def attach_features_to_graph() -> None:
    """
    實戰：將使用者的年齡與薪水，綁定到 NetworkX 的圖中，並抽取為特徵矩陣 X。
    """
    
    # 🌟 建立一個空白的社交網路
    G = nx.Graph()
    
    # 這裡我們換一種建立圖的方式
    # 1. 加入節點 (同時塞入屬性/特徵字典)
    # 我們假設特徵有兩個維度: D=2 (年齡Age, 薪水Salary)
    G.add_node(0, name="Alice", age=25, salary=80000)
    G.add_node(1, name="Bob",   age=42, salary=120000)
    G.add_node(2, name="Carol", age=21, salary=20000)
    G.add_node(3, name="Dave",  age=35, salary=95000)
    
    # 2. 加入邊 (他們是朋友)
    G.add_edges_from([(0, 1), (0, 2), (1, 3)])
    
    # --- 探測工程 ---
    print("--- 節點 0 (Alice) 的內部屬性 ---")
    # G.nodes[0] 會取出一個 dict
    print(G.nodes[0])
    
    print("\n--- 🧠 準備餵給神經網路：萃取特徵矩陣 X ---")
    
    # 初始化一個空的 List 來存放向量
    X_list = []
    
    # 遍歷所有的 N 個節點
    for node_id in G.nodes:
        # 抽取我們需要的數值特徵 (必須丟棄字串 'name')
        age = G.nodes[node_id]['age']
        salary = G.nodes[node_id]['salary']
        
        # 組裝成一個向量 (List) D=2
        node_features = [age, salary]
        X_list.append(node_features)
        
    # 將 List 轉換為神經網路最愛的 numpy 陣列 (N x D)
    X = np.array(X_list, dtype=np.float32)
    
    num_nodes, num_features = X.shape
    
    print(f"特徵矩陣 X 的形狀 (Shape): {X.shape}")
    print(f"N (節點數量) = {num_nodes}")
    print(f"D (特徵維度) = {num_features}")
    print("\n矩陣 X 內容:")
    print(X)
    
    # 至此，我們已經完美準備好可以送入 PyTorch 的資料了！

if __name__ == "__main__":
    attach_features_to_graph()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **特徵尺度不一 (Feature Scaling / Normalization)：** 看看上面印出來的矩陣 $\mathbf{X}$，年齡大約 20~40 歲，但薪水卻高達 100,000 以上。在訓練神經網路時，薪水的「梯度 (Gradient)」跟數字大小會完全輾壓年齡的影響力，導致網路「瞎掉」，只理會薪水。工程師在萃取出矩陣 $\mathbf{X}$ 後，通常必定要接一個 `StandardScaler`（把資料減掉平均值除以標準差），讓所有的特徵都座落在 -1 到 1 的範圍之內。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
如果一個圖有 1,000,000 個節點 ($N = 10^6$)，每個節點的特徵是從一種名為 BERT 的大型語言模型萃取出來的文字語意向量，這個向量的維度是 768 ($D = 768$)。
在 Float32（每個數字佔據 4 Bytes）的資料格式下，請估算這個特徵矩陣 $\mathbf{X}$ 會佔用多少 MB 或 GB 的記憶體空間？

*(提示：記憶體 Bytes = $N \times D \times 4$)*

### Extreme Edge-Case (魔王挑戰)
你現在是一家電商的主管。我們有了商品節點的特徵矩陣 $\mathbf{X}$ (包含商品的長寬高、價格、重量)。
如果我們現在要加入「邊緣特徵 (Edge Features)」呢？（也就是連線上也有屬性，例如兩個人相識的時間長短，或是顧客買這兩個商品的「時間差」只有 5 分鐘）。
請用數學或矩陣的概念思考：你會怎麼表示邊緣特徵 $\mathbf{E_{attr}}$？如果這是一個 $N \times N$ 的圖，且邊緣特徵的維度是 $C$（例如有 3 種不同定義的關係），這個矩陣或張量的形狀 (Shape) 會變成多大？為什麼在現實中 GNN 論文很少探討超高維度的邊緣特徵？