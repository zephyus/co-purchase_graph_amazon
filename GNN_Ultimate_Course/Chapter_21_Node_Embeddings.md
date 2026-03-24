# 第二十一章：把圖壓縮成向量（DeepWalk 與 Node2Vec）

## 1. The Intuition (引言與核心靈魂)

如果我們要把「A認識B」這種無窮無盡的網路連線，直接塞進一個神經網路（像深度學習的 Multi-Layer Perceptron），我們會遇到毀滅性的打擊：神經網路的輸入層（Input Layer）必須是固定長度的向量，但圖的大小是百萬級別，且每個節點連線數量都不一樣。

在 2013 年，NLP (自然語言處理) 領域橫空出世了一個革命性的發明 **Word2Vec**。它能把任何一個「英文單字」壓縮成一個固定長度（例如說 $D=128$）的浮點數向量（Embedding）。
圖論學家在 2014 年靈光一閃：「等等！如果我們也能把「節點」當作「單字」來處理呢？」這就是 **DeepWalk** 與 **Node2Vec** 的誕生，它們正式開啟了 Graph Embedding (圖嵌入) 的黃金時代。這技術把死板的連線結構 $\mathbf{A}$，直接印在了一串數字上。如果兩個節點在圖上經常連在一起，他們在空間中的 Embedding 向量就會非常靠近。

### Learning Objectives (學習目標)
1. **理解 Random Walk (隨機遊走)**：如何把 2D 的圖降維成 1D 的「句子」。
2. **掌握 Node Embedding 的本質**：見證無數個 Float32 數字如何儲存社交網路的人脈資訊。
3. **區別 DeepWalk 與 Node2Vec**：看懂 BFS (廣度優先) 與 DFS (深度優先) 的隨機遊走策略。

---

## 2. Deep Dive (核心概念與深度解析)

### 第一步：Random Walk (隨機遊走)
要把圖 $\mathcal{G}$ 變成像文章一樣的東西，我們讓一個酒醉的虛擬小人站在某個節點上。
小人每一步都「隨機」挑選眼前相連的一條路走過去。
這段路徑就成為了一個「句子」。
例如：從 $v_3$ 出發 $\rightarrow v_8 \rightarrow v_{15} \rightarrow v_2 \rightarrow v_3$。
在這裡，「節點就是單字，Random Walk 走出來的路徑就是句子」。

### 第二步：Word2Vec (Skip-Gram 訓練)
有了幾萬個隨機走路得出來的句子後，我們直接套用 NLP 的演算法 (Skip-Gram)。
它的核心哲學是：「**用中間的單字，去預測它周圍的單字**」。
在圖論裡，這代表「**如果 $v_a$ 和 $v_b$ 經常出現在同一條隨機路徑上（代表他們在圖中距離很近），那麼我就硬把他們在神經網路裡的向量矩陣拉在一起**」。
訓練完之後，每個節點 $v_i$ 就擁有了一個 $\mathbb{R}^D$ 的專屬向量 $\mathbf{Z_i}$。

### Node2Vec: 隨機遊走的進化 (p, q 參數)
DeepWalk 是徹底的「瞎子走路（完全隨機）」。史丹佛大學的 Jure Leskovec 教授發明了 **Node2Vec**，加入了兩個可以人為控制的偏誤參數：
*   **$p$ (Return Parameter)**：控制小人「往回走（回到上一步）」的機率。如果 $p$ 很低，小人就會在原地打轉，這極度適合用來探索 **BFS (廣度優先 search)**，這能捕捉到「同一個小型交友圈」的局部社群特徵 (Local Homophily)。
*   **$q$ (In-out Parameter)**：控制小人「往外瘋狂探索新世界」的機率。如果 $q$ 很低，小人會瘋狂遠離出發點，這適合探索 **DFS (深度優先 search)**，這能捕捉到「不同社群之間的橋樑型人物」。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要訓練出這些 Embeddings，就等於我們跑了圖神經網路 (GNN) 算特徵了。**
  * *真相*：DeepWalk 和 Node2Vec 是**「淺層神經網路 (Shallow Embeddings)」**（或者被稱為 Transductive Learning）。這帶來了一個毀滅性的缺點：如果圖上突然新增了一個全新的節點（例如 Amazon 上架了新書），Node2Vec 是**完全無法即時處理**的。因為它沒有學過這本書的向量！你必須把整張百萬節點的圖重新跑一次 Random Walk 並重新訓練幾個小時，新書才會有 Embedding。這是 Node2Vec 後來被真正的 GNN 淘汰的致命傷。真正的 GNN 學的是「如何傳遞」，而不是死記硬背專屬 ID 向量。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們不會在這裡實作龐大的 Word2Vec 引擎，但我們會寫一個純粹的「Random Walk Generator」，這是 Node2Vec 的核心資料準備階段。

```python
import networkx as nx
import random
from typing import List

def generate_random_walks() -> None:
    """
    實作圖上的無偏隨機遊走 (DeepWalk 風格)，將 Graph 降維成 NLP 訓練語料。
    """
    
    # 🌟 建立一個微型社交網
    G = nx.karate_club_graph() # 經典的空手道俱樂部圖 (34個節點)
    print(f"圖形載入完畢，共有 {G.number_of_nodes()} 個節點")
    
    # 參數設定
    WALK_LENGTH = 10        # 每一步要走多遠 (一個句子的長度)
    NUM_WALKS_PER_NODE = 5  # 每個節點要當作起點出發幾次 (增加資料集大小)
    
    walks: List[List[str]] = []
    
    # 針對圖裡的「每一個節點」都當一次起點
    for node in G.nodes():
        for _ in range(NUM_WALKS_PER_NODE):
            
            # --- 單次隨機遊走的開始 ---
            current_node = node
            walk = [str(current_node)] # 把節點轉換成字串 (模擬單字)
            
            # 開始往前走 `WALK_LENGTH` 步
            for step in range(WALK_LENGTH - 1):
                # 取得當前節點的所有鄰居
                neighbors = list(G.neighbors(current_node))
                
                # 如果這是一個黑洞節點(沒有鄰居)，就只能中斷 (但在 Karate圖裡沒有這種點)
                if not neighbors:
                    break
                
                # 🎲 瞎子摸象：隨機亂選一條相連的路徑
                next_node = random.choice(neighbors)
                walk.append(str(next_node))
                current_node = next_node
                
            # 儲存這條練成出來的「句子」
            walks.append(walk)
            
    print("\n--- 隨機遊走 (Random Walk) 抽樣完成 ---")
    print(f"從 34 個節點中，總共生成了 {len(walks)} 個「圖論句子」。")
    
    print("\n[範例] 前面的三條 Random Walk 路徑 (這就是 Word2Vec 要吃的語料庫):")
    for i in range(3):
        print(f"路徑 {i+1}: {' -> '.join(walks[i])}")

if __name__ == "__main__":
    generate_random_walks()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **Out-of-Memory (記憶體爆炸)：** Random Walk 生出來的句子非常恐怖。假設你有 $N=10^6$ (一百萬個節點)，每個節點出發走 10 次 (`NUM=10`)，長度走到 80 步 (`LENGTH=80`)，你會生出 1000 萬個長度為 80 的句子清單。如果你用 Python 內建的 List 和 String 來硬存，你的記憶體會瞬間爆掉 60GB 以上。所以我們通常寫 C++ (如 PyG 裡的 cluster.random_walk) 直接輸出到硬碟檔案，或者使用 generator 即時生成，絕對不放入一個超大的 `walks_list` 裡。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
考慮兩個不同的商業場景：
A. **Amazon 推薦系統**：我們希望推薦「你可能會喜歡的其他相似類型商品」(例如：買了睡袋，推薦枕頭)。
B. **Twitter 重大新聞散播網**：我們希望找到哪個人是「能夠跨越政治派系，把左派和右派網路連接起來的橋樑」。

請問在 Node2Vec 的 $p$ 和 $q$ 選擇上，這兩個場景分別該設定成 $p, q$ 何者極大？何者極小？
*提示：誰需要 BFS(原地探索小圈圈)？誰需要 DFS(向外長途跋涉)？*

### Extreme Edge-Case (魔王挑戰)
你接手了一個遺留的 Node2Vec 系統。
你發現這個模型給出的 Embeddings 有一種病態的現象：無論什麼節點（例如一本極度冷門的哲學書 $V_a$ 和一本滿街都是的哈利波特 $V_h$），在向量空間中的 Cosine 相似度竟然超級高！
請從 Random Walk 資料生成的角度反思：如果這個網路中存在一群「超級節點 (Super Hub)」（例如：衛生紙，它跟幾乎所有商品都有連線），當 Random Walk 小人隨便亂走的時候，這會導致這些「句子」裡充滿了什麼東西？這又會如何毀滅整個 Word2Vec 的語意空間？
(這就是 Graph Embedding 最著名的 **Hub-Dominance 污染**，也是之後我們在算 Attention 時必須極力避免的死穴)。