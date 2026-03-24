# Amazon Co-Purchase Graph 終極教科書（30 堂）

這是一份給「完全零基礎」學習者的完整教材。
目標是讓你從不會 Python、不懂機器學習，一路走到能完全理解並解釋本專案的圖神經網路流程，特別是 Q4 的進階連結預測。

## 這份教材的特色

- 從零開始，不假設任何背景。
- 每堂課固定包含四個區塊：
  - 核心觀念
  - 推導與直覺
  - 可直接執行的小範例
  - 課後作業
- 所有概念都會回扣到你的實際專案檔案。

## 課程地圖（6 大階段，30 堂）

1. Phase 1: Python 與資料處理基礎（Lesson 1-5）
2. Phase 2: 機器學習核心觀念（Lesson 6-10）
3. Phase 3: 深度學習與 PyTorch（Lesson 11-15）
4. Phase 4: 圖論與 Amazon 圖資料（Lesson 16-20）
5. Phase 5: GNN、PyG 與 GAT（Lesson 21-25）
6. Phase 6: 專案實戰與最終優化（Lesson 26-30）

## 檔案索引

- `PHASE_1_PYTHON_ZERO_TO_ONE.md`
- `PHASE_2_ML_FOUNDATIONS.md`
- `PHASE_3_DEEP_LEARNING_PYTORCH.md`
- `PHASE_4_GRAPH_THEORY_AND_DATA.md`
- `PHASE_5_GNN_PYG_GAT.md`
- `PHASE_6_AMAZON_PROJECT_MASTERY.md`

## 建議學習方式

1. 每天 1 堂課，先讀觀念，再跑範例，再做作業。
2. 所有範例都先在本機執行，再改 1-2 行觀察結果。
3. 每個 Phase 結束後，寫一頁自己的「白話筆記」。
4. 第 30 堂完成後，重新閱讀 `q4_advanced_link_prediction.py`，你會發現每一段都看得懂。

## 最小環境準備

在專案根目錄執行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 你會得到什麼能力

- 能讀懂並修改 Python 與 PyTorch 程式。
- 能解釋 AUC、F1、閾值選擇與資料切分。
- 能說明圖神經網路為何適合 co-purchase graph。
- 能完整講解本專案 Q1-Q4 的技術與結果可信度。

---

如果你已經準備好，請從 Phase 1 Lesson 1 開始。
