# Amazon Co-Purchase Graph 新手完整教學（從 0 開始）

這份文件是給「第一次做機器學習 / 圖神經網路」的人。
你不用先懂 GNN，也可以從頭理解這個專案做了什麼、為什麼這樣做、以及如何重現結果。

---

## 1. 先用一句話理解這個專案

我們把「商品一起被購買」看成一張圖（Graph）：

- 每個商品 = 一個節點（node）
- 兩商品常被一起買 = 一條邊（edge）
- 每個商品有文字評論轉成的數值特徵（767 維）
- 每個商品有類別標籤（10 類）

這個作業分四題：

1. Q1：先把圖建起來，做統計分析
2. Q2：用 GAT 做節點分類（預測商品類別）
3. Q3：改成連結預測（預測未來共購關係）
4. Q4：用更進階模型把連結預測分數往上拉

---

## 2. 你需要先知道的最少背景

### 2.1 什麼是 Graph（圖）

一般機器學習資料是表格：每列獨立。
Graph 資料不同，因為每筆資料（節點）彼此有連線關係（邊）。

在這題裡，關係很重要：

- 單看商品文字特徵可以分類
- 再加上「和哪些商品共購」的關係，通常能做得更好

### 2.2 什麼是 GAT

GAT = Graph Attention Network。

核心想法：

- 每個節點更新自己的表示（embedding）時，不是平均鄰居，而是對不同鄰居給不同權重
- 權重由 attention 機制學出來

直觀上：

- 有些鄰居對你很重要（高權重）
- 有些鄰居訊息雜訊大（低權重）

### 2.3 什麼是 AUC 和 F1

這兩個是 Q3/Q4 連結預測最重要指標。

- AUC：看模型把正樣本排在負樣本前面的能力（排序能力）
- F1：同時考慮 precision 與 recall 的平衡

注意：AUC 高，不代表 F1 一定高。因為 F1 很受分類閾值（threshold）影響。

---

## 3. 資料檔案說明（本專案用到的）

### 3.1 [Dataset/nodes.csv](nodes.csv)

每列一個商品節點，主要欄位：

- node_id：節點 ID
- 0~766：共 767 個特徵欄
- label：類別（0~9）

### 3.2 [Dataset/edges.csv](edges.csv)

每列一條邊：

- source：來源節點
- target：目標節點

此專案把它視為無向圖（undirected）。

### 3.3 [Dataset/classes.csv](classes.csv)

類別編號對應名稱，例如：

- 0 = Desktops
- 2 = Laptops
- 4 = Computer Components

---

## 4. 專案程式結構（你會用到哪些檔）

### 4.1 共用工具

- [Dataset/utils_graph.py](utils_graph.py)
  - 讀資料
  - 建 edge index
  - 做切分
  - 指標計算（AUC/F1）
  - 畫圖與輸出 JSON

- [Dataset/graph_models.py](graph_models.py)
  - GAT 層
  - 節點分類模型
  - 連結預測 encoder + decoder
  - 進階模型元件（Q4）

### 4.2 四題腳本

- [Dataset/q1_graph_stats.py](q1_graph_stats.py)
- [Dataset/q2_gat_node_classification.py](q2_gat_node_classification.py)
- [Dataset/q3_link_prediction.py](q3_link_prediction.py)
- [Dataset/q4_advanced_link_prediction.py](q4_advanced_link_prediction.py)

### 4.3 一鍵執行

- [Dataset/run_all.sh](run_all.sh)

---

## 5. 從零開始跑（實際命令）

在 [Dataset](.) 目錄下執行。

### 5.1 Q1：圖統計

```bash
/home/russell512/.venv/bin/python q1_graph_stats.py --dataset-dir . --output-dir results/q1
```

### 5.2 Q2：GAT 節點分類（35/25/40）

```bash
/home/russell512/.venv/bin/python q2_gat_node_classification.py \
  --dataset-dir . \
  --output-dir results/q2 \
  --train-ratio 0.35 --val-ratio 0.25 --test-ratio 0.40 \
  --device cuda
```

### 5.3 Q3：未來共購預測（temporal proxy）

```bash
/home/russell512/.venv/bin/python q3_link_prediction.py \
  --dataset-dir . \
  --output-dir results/q3 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --device cuda
```

### 5.4 Q4：進階單模型（建議先 cuda，不夠再 cpu）

```bash
/home/russell512/.venv/bin/python q4_advanced_link_prediction.py \
  --dataset-dir . \
  --output-dir results/q4_final_push \
  --device cuda
```

---

## 6. Q1 到 Q4 每題到底在做什麼（白話）

## 6.1 Q1：先把資料健康檢查做完

Q1 不是只要「算數字」，而是要確認資料是否合理。

Q1 會輸出：

- 節點數、邊數、類別數
- 度數分佈（degree distribution）
- 圖密度（density）
- 連通分量（connected components）
- 特徵稀疏度（feature sparsity）

你要看什麼：

- 若有大量孤立節點（isolated nodes），代表連結訊息不足
- 若圖非常稀疏，連結預測通常比較難

## 6.2 Q2：節點分類（Node Classification）

目標：給每個商品預測類別。

流程：

1. 用 stratified split 把節點分成 train/val/test = 35/25/40
2. 用 train 訓練 GAT
3. 用 val 挑最佳 epoch（early stopping）
4. 用 test 報最終表現

輸出重點：

- training_curves.png（loss/acc 是否收斂）
- embedding_before_training.png / embedding_after_training.png
- q2_metrics.json（最終指標）

如何判斷模型有學到：

- train loss 下降
- val acc 上升且不崩壞
- 訓練後 embedding 不同類別更分群

## 6.3 Q3：連結預測（Link Prediction）

目標：預測兩個商品之間是否會有「未來共購邊」。

這份資料沒有時間戳，因此採用 temporal proxy：

- 以 edges.csv 的列順序當作時間順序
- 前段邊當訓練，後段邊當驗證與測試

這樣的好處：

- 符合作業要求「future co-purchase」概念
- 避免訓練看到太多未來訊息

Loss 為什麼用 BCEWithLogitsLoss：

- 這是二元分類最常見且穩定的選擇
- 不需手動先 sigmoid，數值更穩定

Q3 的難點：

- 負樣本怎麼抽（不是邊的節點對）
- threshold 怎麼選（影響 F1 很大）

## 6.4 Q4：進階模型提升

Q4 的策略是：

- 更強的 encoder（residual + 多頭注意力）
- 更強的 decoder（MLP，使用多種邊特徵組合）
- hard negative sampling
- 超參數搜尋（多 trial）

實驗結果中，AUC 已明顯突破 baseline；F1 接近門檻但略低。

---

## 7. 如何閱讀結果檔（不懂也能看）

### 7.1 [Dataset/results/q1/q1_stats.json](results/q1/q1_stats.json)

先看：

- num_nodes / num_edges
- degree.mean
- feature_sparsity.global_sparsity

### 7.2 [Dataset/results/q2/q2_metrics.json](results/q2/q2_metrics.json)

最重要：

- best_val_acc
- test_acc

如果 test_acc 很高，表示節點分類成功。

### 7.3 [Dataset/results/q3/q3_metrics.json](results/q3/q3_metrics.json)

最重要：

- test.auc
- test.f1
- threshold_selected_on_val

### 7.4 [Dataset/results/q4_final_push/q4_metrics.json](results/q4_final_push/q4_metrics.json)

最重要：

- test.auc（是否 >= 0.875）
- test.f1（是否 >= 0.850）
- baseline_targets.pass_*（程式已自動判斷）

---

## 8. 為什麼 AUC 可以很高，但 F1 還是不到 0.85？

這是很多新手最常困惑的點。

原因通常是：

1. 模型排序能力強（AUC 高）
   - 正樣本大多排在前面
2. 但要切成 0/1 時，precision-recall 仍難兼顧
   - threshold 一改，recall 上去但 precision 下來

因此「AUC 達標 + F1 未達標」是合理可能發生的情況，不是程式壞掉。

---

## 9. 常見問題（新手最容易卡）

### 9.1 CUDA OOM（顯存不足）

現象：

- 報錯 out of memory

處理：

- 先改 `--device cpu` 保證能跑完
- 或調小 hidden_dim、heads、embed_dim、neg_ratio
- 或減少 trial 數量

### 9.2 pandas / matplotlib 沒裝

安裝：

```bash
/home/russell512/.venv/bin/pip install -r requirements.txt
```

### 9.3 看不懂圖

先看趨勢：

- loss 下降 = 有在學
- val 指標上升 = 泛化改善
- 後期 val 指標下降 = 過擬合，應早停

---

## 10. 如果你還想再提升 F1，下一步怎麼做

可嘗試：

1. 更精細的 threshold policy
   - 例如在 validation 上做分段門檻搜尋
2. 改負樣本策略
   - 增加 hard negative 比例，但避免過難導致不穩
3. 改 decoder 特徵組合
   - 加入 degree / 共鄰居統計作為額外特徵
4. 做種子平均
   - 換多個 seed 減少偶然波動

---

## 11. 最後給 ML 新手的一句話

你現在已經完成了典型圖機器學習專案的完整流程：

- 資料理解
- 特徵與圖結構分析
- 基線模型
- 任務轉換（分類 -> 連結預測）
- 進階模型優化
- 指標診斷

這就是一個可以放進作品集的完整實戰範例。
