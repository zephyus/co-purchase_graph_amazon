# 第五章：Pandas 神器登場（如何讀取你作業中的 nodes.csv 表格）

## 1. The Intuition (引言與核心靈魂)

如果說 Python 原生的 List 和 Dictionary 是你在家裡隨便拿來裝雜物的各種尺寸收納盒，那麼 **Pandas** 套件就是一整座現代化、自動化、具備 SQL 查詢等級的巨型 Excel 倉庫。

在真實世界的機器學習專案中（如同你最終要解決的 Amazon Co-Purchase Graph），資料絕對不會像課堂範例那樣乾乾淨淨地給一個 Python List：它們會是一大堆以逗號分隔的純文字檔（CSV）、是混雜著缺失值 (Missing Values) 以及字串的髒亂表格。

Pandas 提供了兩個降維打擊的武器：`Series` (單一欄位，一維資料) 與 `DataFrame` (擁有行列的表格，二維資料)。學會 Pandas，你等於學會了不用滑鼠就能在瞬間操縱百萬筆 Excel 資料的黑魔法。

### Learning Objectives (學習目標)
1. **掌握 DataFrame 本質**：理解 Row/Column 的概念與 Pandas 的內部型別對應。
2. **精通 I/O 與基本分析**：學會使用 `pd.read_csv()` 讀取真實資料，並使用 `.head()`, `.describe()`, `.info()` 快速掃描資料健康度。
3. **資料篩選與遮罩 (Boolean Masking)**：放棄用 `for` 迴圈找資料的舊習慣，學會向量化的全表格條件篩選。

---

## 2. Deep Dive (核心概念與深度解析)

為什麼 Pandas 會這麼快？因為它的底層建築在另一個名為 **NumPy** 的 C 語言擴充套件之上。

### 向量化運算 (Vectorization)
在過去，如果我們有一個長度為 $N$ 的數學分數列表，我們想把每個人的分數加 10 分，在純 Python 中我們必須寫：
```python
new_scores = [x + 10 for x in scores]  # Python 層級的迴圈
```
在 Pandas 的世界中，這個運算轉移到了硬體（CPU 暫存器）與 C 語言層級：
$$ \mathbf{X} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad \mathbf{X}' = \mathbf{X} + 10 \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} $$
這被稱為 **廣播機制 (Broadcasting)**。在 Pandas 中你只需要寫 `df['score'] + 10`。因為它在底層不需要進行 Python 動態型別檢查的開銷 (Type Checking Overhead)，運算速度通常可以提升數十倍甚至百倍。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：當你想看表格裡的特定條件資料時，你應該用 `iterrows()` 一行一行跑迴圈去查。**
  * *真相*：在 Pandas 裡面寫 `for` 迴圈是對效能的嚴重侮辱（這被稱為 Pandas Anti-pattern）。所有條件過濾都必須透過「布林遮罩 (Boolean Mask)」以向量化的方式一次完成。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們來模擬你期末專案中最關鍵的第一步：讀取 `nodes.csv` (節點特徵 / 商品資訊/ 論文分類)，並進行資料清洗。

```python
import pandas as pd
import numpy as np

def analyze_nodes_dataset(csv_path: str) -> None:
    """
    示範如何讀取 CSV、檢查資料缺失、以及利用布林遮罩進行向量化篩選。
    這支程式具備完全的防禦機制。
    """
    try:
        # 1. 讀取並建立 DataFrame (DF)
        # 在這裡可以模擬讀取真實路徑，若檔案不存在我們用 catch 機制接住
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到檔案: {csv_path}。我們將動態生成一份模擬的 nodes.csv 來做示範！")
        
        # 動態生成一個帶有特徵與一點點髒資料的 DataFrame
        data = {
            'node_id': [0, 1, 2, 3, 4],
            'label': ['Book', 'Electronic', 'Book', 'Clothing', 'Electronic'],
            'feature_1': [0.55, 0.12, np.nan, 0.88, 0.93], # np.nan 模擬缺失值
            'feature_2': [1.2, 3.4, 0.5, 2.1, 0.1]
        }
        df = pd.DataFrame(data)

    print("\n--- 1. 資料前五筆概覽 ---")
    print(df.head())
    
    print("\n--- 2. 資料集健康度診斷 ---")
    # .info() 能看出哪些欄位有缺值 (Non-Null Count)，這在決定是否要 dropna 或是 fillna 前必做
    df.info()

    # 3. 實作向量化布林遮罩 (Boolean Masking) 代替 For 迴圈
    # 任務：我們只想要抓出 'label' 是 'Book' 的節點，而且不能用 for 迴圈
    print("\n--- 3. 篩選特定類別 (Book) 的節點 ---")
    
    # 建立遮罩：會產生一條 [True, False, True, False, False] 的布林一維陣列
    is_book_mask = (df['label'] == 'Book')
    
    # 將遮罩套套回 DataFrame
    book_nodes_df = df[is_book_mask]
    print(book_nodes_df)
    
    # 4. 缺失值處理 (Handling Missing Data)
    # 由於深度學習模型 (如 PyTorch) 無法吃進含有 NaN (Not a Number) 的 Tensor
    # 我們必須在進模型前把 feature_1 的 NaN 補上平均值
    print("\n--- 4. 將 NaN 填補為平均值 ---")
    mean_f1 = df['feature_1'].mean()
    df['feature_1'] = df['feature_1'].fillna(mean_f1)
    
    # 檢查是否還有任何的 NaN 殘留
    print(df.head())

if __name__ == "__main__":
    # 試圖讀取一個不存在的檔案觸發容錯機制
    analyze_nodes_dataset("dummy_amazon_nodes.csv")
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **記憶體爆炸 (Out of Memory for huge CSVs)：** 當你的 `edges.csv` 高達 50 GB，但你的 RAM 只有 16 GB 時，`pd.read_csv()` 會直接引發系統 OOM 死亡。這時工程解法是加上 `chunksize=100000` 參數，它會回傳一個迭代器，讓你每次只讀 10 萬筆進入記憶體處理，甚至放棄 Pandas 改用 `Dask` 或是 `Polars` 等分散式處理框架。對於你的專案，若發現資料過大，也可跳過 Pandas 直接使用純 C 讀取。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
考慮以下兩個將特定欄位的值翻倍的操作：
1. `df['price'] = df['price'].apply(lambda x: x * 2)`
2. `df['price'] = df['price'] * 2`
在資料量高達一千萬筆時，哪一個速度會快？為什麼？請說明 `apply` 與底層 C 語言廣播機制的差別。

### Extreme Edge-Case (魔王挑戰)
在 Pandas 中，如果你試著執行 `df[df['label'] == 'Book' and df['feature_1'] > 0.5]`，程式會直接報錯 `ValueError: The truth value of a Series is ambiguous`。
請查閱官方文件，為什麼 Python 原生的 `and` 運算子無法用在 Pandas 的 Series 遮罩上？你必須將全域邏輯算子換成哪兩個符號（以實現 element-wise boolean operations）？同時，在語法上為什麼圓括號 `()` 變得絕對必要？