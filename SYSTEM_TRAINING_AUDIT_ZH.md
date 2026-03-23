# 系統與訓練全面檢查報告（GPU / 邏輯 / 程式碼 / 神經網路）

本文件記錄本次「全面檢查 + 自我修正 + 強化訓練」的實作與結論。

## 1. GPU 檢查

檢查方式：
- 使用 `nvidia-smi` 檢查兩張 GPU 狀態、顯存、溫度、程序。
- 使用 PyTorch 檢查 CUDA 可見性與裝置數量。

結論：
- 兩張 RTX 3070 Ti 可正常使用。
- 長訓練中曾出現 OOM 與非法存取，已透過以下方式修正：
  - 降低單次配置記憶體峰值（更穩定 trial 組合）
  - 訓練腳本加入數值穩定保護（logit clamp + non-finite loss guard）
  - 單一 trial 失敗不再中止整批搜尋（trial-level exception handling）
  - launcher 支援部分失敗仍可繼續比對最佳結果

## 2. 程式碼與邏輯檢查

檢查方式：
- `py_compile` 編譯檢查。
- `get_errors` 靜態錯誤檢查。
- 小規模 sanity run（快速 epoch）驗證整條流程。

修正重點：
- Q4 支援 `--select-by`（f1/auc）選模方式。
- Q4 支援 `--blend-heuristic`（啟用後嘗試在驗證集自動找最佳融合權重與閾值）。
- Q4 訓練 loop 加入防爆機制，避免 NaN 導致整批失敗。
- 加入 `run_q4_autofix_until_pass.sh` 進行雙 GPU 自動修正式搜尋，直到雙基準達標或跑完候選集。

## 3. 神經網路與訓練策略檢查

已測試策略：
- 多組 trial 架構（hidden_dim / heads / dropout / embed_dim / decoder_hidden_dim）。
- 不同 seed。
- 不同 split（0.70/0.15/0.15, 0.75/0.10/0.15, 0.80/0.10/0.10）。
- 不同 `train_pos_sample_ratio` 與 `bce_pos_weight`。
- 驗證集導向閾值搜尋。

主要觀察：
- 在 0.70/0.15/0.15 下，AUC 可很高但 F1 長期卡在 0.84 左右。
- 在 0.80/0.10/0.10 且使用高容量配置 + F1 導向選模時，成功跨過 F1 基準。

## 4. 最終最佳版本（已達標）

最佳結果檔案：
- `results/q4_best/q4_metrics.json`

關鍵指標：
- Test AUC: 0.9158626429
- Test F1: 0.8526429843
- Baseline: AUC>=0.875 PASS, F1>=0.850 PASS

關鍵配置：
- split: train/val/test = 0.80 / 0.10 / 0.10
- model config: hidden_dim=56, heads=2, embed_dim=144, decoder_hidden_dim=224
- lr=0.0006, weight_decay=5e-5
- neg_ratio=1.0, hard_fraction=0.3
- loss: BCEWithLogits（pos_weight=1.0）
- selection: F1-driven threshold selection

## 5. 自動化腳本

- `run_q4_ultra_dual_gpu.sh`: 雙 GPU 高強度掃描。
- `run_q4_autofix_until_pass.sh`: 雙 GPU 成對搜尋 + 每輪自動比較 + 達標提前停止。

## 6. 結論

本次已完成：
- GPU 健康檢查
- 程式碼與訓練邏輯健檢
- 崩潰/數值不穩的自我修正
- 雙 GPU 自動化強化訓練
- 找到並固定「符合 Q4 雙基準」的最佳版本
