# 第十九章：連結預測大揭密（什麼是 Link Prediction？）

## 1. The Intuition (引言與核心靈魂)

現在回到你的終極任務：「如果買了這本書，顧客還會買什麼？」
這是一個古老且價值千金的商業命題。Amazon 和 Netflix 利用這項技術賺進了數千億美元。
在圖論的語言裡，這項任務有一個專屬的名詞：**「連結預測 (Link Prediction)」**。

簡單來說，給定一張已經存在著連線 (Edges) 的圖，我們要讓演算法去猜測：**「在未來，哪兩個根本還沒連線的節點之間，最有可能長出一條新的連線？」**
*   **推薦系統 (Amazon/Netflix)**：預測 User $U_i$ 會不會買 Item $I_j$？
*   **社交推薦 (Facebook)**：你可能認識這個人（預測 Person A 和 Person B 會連線）。
*   **藥物研發 (Bioinformatics)**：預測蛋白質 A 會不會對藥物分子 B 產生反應 (Binding)。

### Learning Objectives (學習目標)
1. **問題定義 (Problem Formulation)**：搞懂在圖論中，機器學習要「預測」什麼。
2. **正負樣本 (Positive/Negative Samples)**：見證圖論最大的工程深淵——如何在一張充滿了 0（沒有連線）的圖中，挑選那些神經網路真正該學的資料。
3. **基礎特徵 heuristic heuristics**：學會怎麼在不用任何神經網路的情況下，光靠「共同朋友有幾個」這件事情做到極度暴力的神準預測。

---

## 2. Deep Dive (核心概念與深度解析)

在我們把整張神經網路搬出來之前（那是在第二十五章的事情），我們必須先釐清 Link Prediction 在做什麼事。

這其實是一個 **二元分類問題 (Binary Classification)**。
我們把所有的「連線」看作是資料點 (Data Points)。
*   **$y = 1$ (Positive Edge / 正樣本)**：這兩個點之間**具有真切的連線**（例如 A 實際買了 B）。
*   **$y = 0$ (Negative Edge / 負樣本)**：這兩個點之間**並沒有連線**（例如 A 這輩子沒聽過 B 這本書）。

### The Negative Sampling Nightmare (負樣本的惡夢)
在你在第十章學到的 Logistic Regression 裡，我們預測貓跟狗，貓的圖片和狗的圖片數量都是平衡的。
但在圖論中，如果你有一個 10,000 個使用者的社交網路，通常一個人平均只認識 100 人。
10,000 個節點能組成的潛在連線數量是 $N(N-1)/2 \approx 50,000,000$ 條。
但是真實存在的連線只有 $10,000 \times 100 / 2 = 500,000$ 條正樣本。
這代表**負樣本 (0) 是正樣本 (1) 的 100 倍以上！** 這是極端不平衡 (Extremely Imbalanced Data)。

如果把所有的 $0$ 都餵給神經網路訓練，模型甚至不用學，它只要閉著眼睛全部猜 $0$，準確率（Accuracy）就高達 99%。這是一場 AI 的災難（這也是為什麼我們之前學的 F1-Score 或 ROC-AUC 才是王道）。
所以在做 Link Prediction 之前，我們工程師都會用 **「隨機負採樣 (Random Negative Sampling)」** 的技巧，強行只抽取和正樣本差不多數量的 0。

### Baseline Heuristics (基準啟發式指標)
在動用 GNN 之前，有很多純數學的傳統方法可以測量 $A$ 和 $B$ 會不會連線。
最經典的想法是：**「如果我們有很多共同朋友，我們很大概率以後也會變成朋友」**。

這種基於鄰居重疊的方法稱為 **Common Neighbors (共同鄰居數)**，或是進階版的 **Jaccard Coefficient (雅卡爾相似係數)**：
$$ \text{Common Neighbors}(A, B) = |\Gamma(A) \cap \Gamma(B)| $$
*這裡的 $\Gamma(X)$ 代表節點 $X$ 所擁有的所有鄰居集合（也就是所有的朋友）。*

如果 Alice ($A$) 認識 5 個人，Bob ($B$) 認識 6 個人，而且他們之中有 4 個人是重複的（交集 $\cap$ 的數量為 4）：
那麼這兩個人的 Common Neighbors 分數就是 4。只要分數超過某個門檻（例如 3），傳統機器學習就會宣判：「推薦他們成為好友！」。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：只要靠「共同好友數量」，我就能打造出超越 Amazon 的天王級推薦演算法。**
  * *真相*：如果 $A$ 是一個買了 5000 種商品（瘋狂購物狂），而 $B$ 是一個只買了 2 種筆記本的窮學生。他們很可能會共同買過「衛生紙」這種大家都買的東西（共同好友 > 1）。但其實他們根本不是同一種人。這就是 Common Neighbors 只看了「交集（有幾個相同的）」，卻忽略了「聯集（他們各自有多少專屬好友）」的硬傷。我們會在下一章引進更暴力的 Jaccard 系數。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們現在要用非常暴力的 Python 原生功能 (Set 交集) 來自己實作 Link Prediction 中最古老的 Common Neighbors 演算法。

```python
import networkx as nx
from typing import Set

def perform_link_prediction_baseline() -> None:
    """
    用 Python 實作最基底的 Link Prediction: Common Neighbors
    """
    
    # 1. 建立一個網路
    G = nx.Graph()
    # 假設我們是一個大學班級
    # Alice 和 Eve, Charlie 是朋友
    # Bob 和 Eve, Charlie 也是朋友
    # 但 Alice 和 Bob 還不認識！
    G.add_edges_from([
        ('Alice', 'Eve'), 
        ('Alice', 'Charlie'),
        ('Bob', 'Eve'),
        ('Bob', 'Charlie'),
        ('Bob', 'David')
    ])
    
    # 2. 假設我們是 Facebook 推薦演算法工程師
    node_A = 'Alice'
    node_B = 'Bob'
    
    # 檢查他們是不是已經認識了 (避免推薦已經存在的好友)
    if G.has_edge(node_A, node_B):
        print(f"[{node_A}] 和 [{node_B}] 已經是朋友了，不需要預測！")
        return
        
    print(f"--- 啟動 {node_A} 推薦機制 ---")
    
    # 取出他們各自的「朋友圈 (Neighbors)」
    # 🌟 [工程心法]: set() 是一種非常暴力的資料結構，拿來做交集運算極快
    neighbors_A: Set[str] = set(G.neighbors(node_A))
    neighbors_B: Set[str] = set(G.neighbors(node_B))
    
    print(f"{node_A} 的朋友: {neighbors_A}")
    print(f"{node_B} 的朋友: {neighbors_B}")
    
    # 3. 計算 Common Neighbors (數學上的交集)
    common_friends = neighbors_A.intersection(neighbors_B) # A ∩ B
    num_common_friends = len(common_friends)
    
    print("\n--- 🧠 演算法分析結果 ---")
    print(f"共同好友列表: {common_friends}")
    print(f"他們的 Common Neighbors 得分是: {num_common_friends} 分")
    
    # 4. 決策門檻 (Thresholding)
    THRESHOLD = 1
    if num_common_friends > THRESHOLD:
        print(f"🔥 [推薦發送] 發現高潛力未連接邊！立刻將 {node_B} 推薦給 {node_A}！")
    else:
        print(f"🧊 [推薦拒絕] 他們交集太少了，不太可能變成好友。")

if __name__ == "__main__":
    perform_link_prediction_baseline()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **資料穿越 (Data Leakage)：** 當你在做 Link Prediction 神經網路訓練時（這是新手最容易犯的低級錯誤），你會拿圖 $\mathcal{G_{train}}$ 去訓練，然後用未來的連線 $\mathcal{E_{test}}$ 來驗證對不對。但如果你不小心讓神經網路在訓練過程中**「看到了 $\mathcal{E_{test}}$」**（例如：在算鄰接矩陣 $\mathbf{A}$ 的時候忘了扣除未來的連線），神經網路會在測試時拿到 99.99% 的超人準確率（因為它作弊偷看了答案）。這在企業裡被稱為資訊穿越，這會毀掉你整個部門的年終獎金。PyTorch Geometric (PyG) 用盡了各種 `train_test_split_edges` 魔法來防止這件事發生，我們之後會遇到。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
有兩個大學生 A 和 B。他們同班，因此互相擁有一群共同的好友。
這座校園中也有兩個天王級別的大企業家 C 和 D（都是這個商學院的校友）。因為大企業家人脈極廣（Degree 極大！），他們認識很多人，剛好他們之間**有 3 個共同朋友**（分別是總統、某教授和某市長）。但是 A 和 B 的共同朋友也是 **3 個**（他們班上的三個同學）。

如果你只用 **Common Neighbors = 3** 這個分數來做預測，系統會給 (A, B) 和 (C, D) 一樣強烈的推薦程度。
請思考一下，這合理嗎？
在現實中，C和D的 3 個交集，佔了他們一萬個人的朋友圈中的多少比例？而 A 和 B 的 3 個交集，又佔了他們五個朋友的多少比例？
這個極度不平衡的盲點可以怎麼補救？

### Extreme Edge-Case (魔王挑戰)
你現在想要預測一個蛋白質網絡 (Protein-Protein Interaction Network) 裡面，節點 X 和 Y 會不會連結 (Binding)。
問題是，在化學分子的世界裡，有些化學鍵結是**「異質相斥」**的：正極只會和負極相連，而不是和另一個正極相連。在這種圖結構中（我們稱之為 Bipartite Graph 或者具有 Heterophily 反同質性的圖），X 和 Y 之間**絕對不允許有共同的鄰居**。
請問：如果我們在這種圖上強行使用「Common Neighbors = 共同朋友數量要高」作為連線預測基準，會發生什麼慘劇？這證明了僅依賴圖形結構（沒有利用特徵矩陣 $\mathbf{X}$ 裡面的物理性質）本身有什麼巨大的缺陷？