# Hướng Dẫn Thực Nghiệm: Consequence-Weighted On-Policy Distillation (CW-OPD)

Tài liệu này cung cấp toàn bộ hướng dẫn cài đặt, cấu trúc code và các bash script để chạy thực nghiệm theo đề cương nghiên cứu:
**"Where Does the Student Lose the Teacher's Conclusion? Teacher Answer-Belief Shifts as a Verifier-Free Allocation Signal for On-Policy Distillation" (Version 7)**.

---

## 1. Tổng Quan Kiến Trúc & Các File Đã Xây Dựng

Hệ thống được tích hợp trực tiếp vào framework **veRL** (Volcano Engine Reinforcement Learning):

```
EOPD/
├── verl/trainer/ppo/cw_opd/               # Core Algorithm
│   ├── candidate_manager.py              # C_x = {a_T} U H_x, canonicalization, symbolic check
│   ├── probe_scorer.py                   # Paragraph segmentation, q_k(a), q_k(⊥), TV shift g_k, w_tilde_t
│   ├── policy_loss.py                    # Loss cw_opd, cw_eopd, fuw (PPO surrogate + forward-KL Soft KD)
│   ├── cw_opd_trainer.py                 # CWOPDTrainer (kế thừa OnPolicyDistillTrainer)
│   └── __init__.py
├── phase0/                               # Phase 0: Pilot Without Training
│   └── run_phase0_pilot.py               # P1 (Calibration, Delta R^2, Clustered Bootstrap), P2, P3, Decision Rule
├── eval/                                 # Evaluation Suite
│   └── evaluate_benchmarks.py            # Qwen2.5-Math protocol (8 samples, Avg@8, Pass@8, Paired Bootstrap)
├── scripts/                              # Bash Scripts chạy trên Server
│   ├── run_all_experiments_master.sh     # Master pipeline script điều phối toàn bộ
│   ├── run_cache_teacher.sh              # Bước tiền xử lý: Cache đáp án greedy của teacher a_T
│   ├── run_phase0_pilot.sh               # Chạy thử nghiệm Phase 0 Pilot
│   ├── run_train_t1_opd.sh               # Table 1: Baseline 1 - Standard OPD
│   ├── run_train_t1_eopd.sh              # Table 1: Baseline 2 - EOPD (Jin et al., ICML 2026)
│   ├── run_train_t1_fuw.sh               # Table 1: Baseline 3 - FUW (Token utility)
│   ├── run_train_t1_ours_cw_opd.sh       # Table 1: Method 4 - Ours (CW-OPD)
│   ├── run_train_t1_ours_cw_eopd.sh      # Table 1: Method 5 - Ours + EOPD (CW-EOPD)
│   ├── run_train_t1_permutation.sh       # Table 1: Control 6 - Permutation Control
│   ├── run_train_t2_ablation_at_only.sh  # Table 2: Ablation (a) - C_x = {a_T} Only
│   ├── run_train_t2_ablation_length_norm.sh # Table 2: Ablation (b) - Length-Normalized Shift g_k / L_k
│   ├── run_train_t2_ablation_kl.sh       # Table 2: Ablation (c) - KL Information-Gain Shift
│   ├── run_eval_all_benchmarks.sh        # Đánh giá checkpoint trên 6 benchmarks
│   ├── cache_teacher_answers.py          # Python runner caching a_T
│   └── prepare_eval_benchmarks.py        # Tải & chuyển đổi datasets test sang parquet
└── tests/
    └── test_cw_opd.py                    # 9 Unit tests kiểm tra toàn bộ logic toán & gradient
```

---

## 2. Chuẩn Bị Môi Trường Server

Trước khi chạy, hãy kích hoạt môi trường conda chứa CUDA và PyTorch trên server:

```bash
cd /home/luyenv/dev/EOPD
pip install -r requirements.txt
```

Kiểm tra trạng thái hệ thống:
```bash
bash scripts/run_all_experiments_master.sh check
```
*(Lệnh này sẽ kiểm tra GPU CUDA, import các thư viện và chạy tự động bộ 9 unit tests).*

---

## 3. Quy Trình Chạy Thực Nghiệm Từng Bước

### Bước 1: Chuẩn bị dữ liệu (Data Preparation)
Chuẩn bị tập train MATH và các tập benchmark (MATH500, AIME24, AIME25, AMC23, Minerva, OlympiadBench):
```bash
bash scripts/run_all_experiments_master.sh data
```

### Bước 2: Cache trước đáp án của Teacher ($a_T$)
Theo Mục 3.2 của đề cương, teacher giải trước mỗi câu một lần (greedy) để cố định $a_T$:
```bash
bash scripts/run_cache_teacher.sh
```
*(Kết quả lưu tại `data/math/teacher_answers_qwen3_8b.json`).*

### Bước 3: Chạy Phase 0 Pilot (Pilot Without Training)
Thực hiện Mục 5 của proposal để kiểm chứng tín hiệu probed shift trước khi train:
* Adaptive sampling continuations ($M=8 \to 32$).
* Ước lượng $e(h) = \mathrm{TV}(q(h), \alpha_{\mathcal{P}}(h))$.
* Hồi quy $\hat{g}^{\mathcal{P}}_k$ trên signals địa phương và probed shift $g_k$, tính $\Delta R^2$, Partial Spearman rank $\rho$ và Bootstrap 90% CI.
* Kiểm tra martingale gap và phát hiện bước mất đáp án ($k^*, k^-$) trước khi xuất hiện đáp án.
* Đo overhead runtime (mục tiêu $\le 25\%$).
* **Tự động đưa ra kết luận Go/No-Go Decision**:

```bash
bash scripts/run_phase0_pilot.sh
```
Báo cáo markdown chi tiết sẽ được tự động tạo tại `experiments/phase0_pilot/phase0_report.md`.

---

## 4. Chạy Các Thực Nghiệm Huấn Luyện (Training)

Mỗi mô hình được train theo cấu hình chuẩn trong Section 6:
* **Teacher**: Qwen3-8B (non-thinking)
* **Student**: Qwen3-1.7B-Base
* **Hardware**: Mặc định 4 GPU A100 (có thể tùy chỉnh biến `N_GPUS=4` hoặc `N_GPUS=8`)
* **Learning Rate**: $3 \times 10^{-6}$
* **Batch size**: 128 prompts (mini-batch 32 $\to$ 4 mini-batch steps/rollout)
* **Max Response Length**: 4096 tokens, Nhiệt độ rollout = 1.0, 3 epochs.

### Bảng 1: Các Thực Nghiệm So Sánh Chính (Table 1)
```bash
# 1. OPD Baseline
bash scripts/run_train_t1_opd.sh

# 2. EOPD Baseline (Jin et al., ICML 2026: tau=0.8, coef=1.0, top-16)
bash scripts/run_train_t1_eopd.sh

# 3. FUW Baseline (Token utility trust-region)
bash scripts/run_train_t1_fuw.sh

# 4. Phương Pháp Đề Xuất (Ours: CW-OPD)
bash scripts/run_train_t1_ours_cw_opd.sh

# 5. Phương Pháp Đề Xuất Kết Hợp EOPD (Ours + EOPD)
bash scripts/run_train_t1_ours_cw_eopd.sh

# 6. Permutation Control (Xáo trộn g_k theo tầng vị trí & độ dài)
bash scripts/run_train_t1_permutation.sh
```

### Bảng 2: Các Thử Nghiệm Tách Biệt Biến (Table 2 - Ablations)
```bash
# (a) Chỉ dùng C_x = {a_T} (bỏ lịch sử sinh câu trả lời trước đó)
bash scripts/run_train_t2_ablation_at_only.sh

# (b) Phân bổ chuẩn hóa độ dài g_k / L_k
bash scripts/run_train_t2_ablation_length_norm.sh

# (c) Biến thể Information-Gain KL(q_k || q_{k-1})
bash scripts/run_train_t2_ablation_kl.sh
```

---

## 5. Đánh Giá Trên Các Benchmark (Evaluation)

Đánh giá checkpoint đã train theo giao thức chuẩn **Qwen2.5-Math** (8 samples, temperature 1.0, top-p 0.8, 8192 tokens max, tính Avg@8 và Pass@8 với Paired Bootstrap):

```bash
# Đánh giá một checkpoint cụ thể:
bash scripts/run_eval_all_benchmarks.sh checkpoints/t1_ours_cw_opd/global_step_final

# Hoặc đánh giá toàn bộ các checkpoints đã train:
bash scripts/run_all_experiments_master.sh eval
```

Các benchmark được đánh giá bao gồm:
1. `MATH500`
2. `AIME24`
3. `AIME25`
4. `AMC23`
5. `Minerva Math`
6. `OlympiadBench`

---

## 6. Chạy Toàn Bộ Tự Động (Master Pipeline)

Bạn có thể chạy toàn bộ chu trình từ đầu đến cuối chỉ với một lệnh:
```bash
bash scripts/run_all_experiments_master.sh all
```
Hoặc chạy từng giai đoạn độc lập:
* `bash scripts/run_all_experiments_master.sh check`: Kiểm tra môi trường & test.
* `bash scripts/run_all_experiments_master.sh data`: Tải & xử lý dữ liệu.
* `bash scripts/run_all_experiments_master.sh cache`: Tạo cache $a_T$.
* `bash scripts/run_all_experiments_master.sh phase0`: Chạy Phase 0 Pilot.
* `bash scripts/run_all_experiments_master.sh table1`: Chạy toàn bộ 6 mô hình Bảng 1.
* `bash scripts/run_all_experiments_master.sh table2`: Chạy toàn bộ 3 ablations Bảng 2.
* `bash scripts/run_all_experiments_master.sh eval`: Đánh giá kết quả trên 6 benchmarks.
