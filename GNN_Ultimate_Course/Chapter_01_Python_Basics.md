# 第一章：哈囉 Python！打通任督二脈（變數與資料型態）

## 1. The Intuition (引言與核心靈魂)

想像你現在是一座巨大圖書館（電腦記憶體）的新任管理員。每天有成千上萬的書籍、卷宗、零碎的紙條被送進來。如果你隨便亂丟，當你需要找昨天的某一筆帳目時，絕對會徹底崩潰。

在 Python 的世界裡，**變數 (Variables)** 就是你貼在各式收納盒上的「標籤」；而**資料型態 (Data Types)** 則是這些盒子的「形狀與材質」。有些箱子專門裝整數（無小數點）、有些裝帶小數點的測量值，而有些則裝一整段的純文字。當你把這些標籤和對應的箱子管理得井井有條時，你就在和電腦進行最流暢的對話。這就是編程的起點。

### Learning Objectives (學習目標)
1. **理解記憶體與變數的關聯**：掌握 Python 如何將資料指派給變數。
2. **精通核心資料型態**：區分整數 (`int`)、浮點數 (`float`)、字串 (`str`) 與布林值 (`bool`)。
3. **掌握基礎運算邏輯**：學會基本的數學運算符號與型態轉換技巧。

---

## 2. Deep Dive (核心概念與深度解析)

在多數低階語言（如 C 語言）中，宣告變數時需要在記憶體中預先切出一塊固定大小的空間。然而，Python 是一種**動態強型別語言 (Dynamically but Strongly Typed)**。這意味著變數本身沒有型態，變數只是一個指向記憶體位址的「參考 (Reference)」，真正擁有型態的是存在記憶體中的那個「物件 (Object)」。

從數學與邏輯的形式化定義來看：
設 $\mathcal{V}$ 為所有合法變數名稱的集合，$\mathcal{O}$ 為所有記憶體中物件的集合。
在 Python 執行賦值運算 $x = 10$ 時，實際上是建立了一個映射函數 $f: \mathcal{V} \rightarrow \mathcal{O}$，使得 $f(x) = \text{Object}(10)$。

如果我們接著執行 $x = \text{"Hello"}$，並不是把字串塞進原本裝數字的洞裡，而是直接將 $f(x)$ 的箭頭改指向一個新的字串物件：$f(x) = \text{Object("Hello")}$。這就是動態型別的數學本質。

### 常見資料型態定義
1. **Integer (`int`)**: 整數集合 $\mathbb{Z}$，如 $x = 5$。
2. **Floating-Point (`float`)**: 實數集合 $\mathbb{R}$ 的有限精度逼近（通常是 IEEE 754 雙精度浮點數），如 $y = 3.14159$。
3. **String (`str`)**: 字元序列 $\Sigma^*$，如 $s = \text{"GNN"}$。
4. **Boolean (`bool`)**: 邏輯值 $\{ \text{True}, \text{False} \}$，對應布林代數。

### 🚨 Common Misconceptions (常見迷思)
* **迷思：變數是一個水桶，裡面裝著資料。** 
  * *真相*：在 Python 中，變數是一張便利貼。當你寫下 `a = b` 時，你只是把 `a` 這張便利貼貼到 `b` 所貼著的同一個物件上，並沒有複製資料本身（這對後續理解記憶體管理極度重要）。

---

## 3. Code & Engineering (程式碼實作與工程解密)

我們現在將上述概念轉化為嚴謹、優雅且具備業界標準（遵循 PEP8，加入 Type Hints）的 Python 實作。

```python
# 導入型別提示庫 (雖然 Python 是動態型別，但在大型專案中 Type Hints 是減少 Bug 的救星)
from typing import Union, List

def demonstrate_variable_binding() -> None:
    """
    展示 Python 中變數的綁定(Binding)與記憶體位址(ID)的變化。
    這有助於理解動態型別語言的底層行為。
    """
    
    # 1. 基礎型態宣告 (結合 Type Hints)
    num_nodes: int = 150  # 圖論中的節點數量
    learning_rate: float = 0.001 # 類神經網路的學習率
    model_name: str = "Graph Attention Network" # 模型名稱
    is_converged: bool = False # 模型是否已收斂
    
    print(f"[{model_name}] 啟動。節點數: {num_nodes}, 學習率: {learning_rate}")
    
    # 2. 記憶體位址與物件綁定 (Reference Binding)
    a: int = 10
    b: int = a
    # a 和 b 此時指向記憶體中的同一個 '10' 物件
    print(f"a 的 ID: {id(a)}, b 的 ID: {id(b)} -> 是否相同？ {id(a) == id(b)}")
    
    # 修改 a 的值 (整數是不可變物件 Immutable，因此 a 會指向新的物件)
    a = 20
    print(f"改變 a 後 -> a 的 ID: {id(a)}, b 的 ID: {id(b)}")
    print(f"此時 b 的值依然是: {b}")

if __name__ == "__main__":
    demonstrate_variable_binding()
```

### 💡 Engineering Edge Cases (工程邊角案例)
* **大整數運算：** 在 C/C++ 中，超過 64-bit 的整數會發生溢位 (Overflow, $x > 2^{63}-1$)。但在 Python 中，`int` 支援任意精度運算。這在深度學習計算中很方便，但也意味著當你需要 GPU 進行張量運算時，必須嚴格將 Python 整數轉換回硬體級別的固定精度（如 PyTorch 的 `torch.int64`）。
* **浮點數精度遺失：** $0.1 + 0.2 \neq 0.3$ 是 IEEE 754 的通病。在深度學習計算 Loss 時，如果數值微小到極限（Underflow），會導致神經網路權重變成 `NaN`。這些我們在後續 PyTorch 章節會用 `log-sum-exp` 等數值穩定技巧來解決。

---

## 4. MIT-Level Exercises (課後思考與魔王挑戰)

### Conceptual Validation (概念驗證)
請不使用電腦，推導以下程式碼的輸出結果，並用「變數是標籤（便利貼）」的模型來解釋為什麼。
```python
x = 100
y = x
x = x + 1
```
請問最後 `y` 的值是多少？`x` 的值是多少？它們在記憶體中還指向同一個空間嗎？

### Extreme Edge-Case (魔王挑戰)
執行這段程式碼：
```python
a = 256
b = 256
print(a is b)  # 判斷記憶體位址是否相同

c = 1000
d = 1000
print(c is d)
```
你會發現第一個輸出 `True`，第二個輸出 `False`。請查閱 Python 官方文件關於 "Small Integer Caching" (小整數快取) 的機制，解釋為什麼在 CPython 直譯器中會發生這種看似矛盾的現象。這對於開發極致效能的程式有什麼啟發？