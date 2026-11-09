# Raw vs. cleaned training data — LM comparison

Fixed budget: **333,724 characters** per condition (= size of the fully curated set); char-level 5-gram Kneser-Ney, shared vocabulary (221 symbols), 3 subsampling seeds (mean ± std). Eval: 240 held-out educational documents (English + Urdu) built from sentences never used in training.

| Training data | Docs available | BPC (all) ↓ | Char PPL (all) ↓ | BPC en ↓ | BPC ur ↓ | Next-char acc ↑ | Δ PPL vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 2,707 | 1.5519 ± 0.0049 | 2.932 | 1.5890 | 1.5133 | 0.6937 | +0.0% |
| +lang | 1,986 | 1.5260 ± 0.0029 | 2.880 | 1.5459 | 1.5051 | 0.7014 | -1.8% |
| +heuristics | 1,514 | 1.4898 ± 0.0034 | 2.808 | 1.5088 | 1.4700 | 0.7086 | -4.2% |
| +dedup | 1,257 | 1.4826 ± 0.0056 | 2.795 | 1.5092 | 1.4549 | 0.7072 | -4.7% |
| full | 678 | 1.4013 | 2.641 | 1.4172 | 1.3847 | 0.7237 | -9.9% |

Raw, unbudgeted (1,018,405 chars, 3.1× more data): BPC 1.5488, char PPL 2.926.

**Tiny GPT** (2 layers, d=64, 300 steps × 32 × 64 chars, identical compute for both):

| Training data | BPC (all) ↓ | Char PPL ↓ | BPC en ↓ | BPC ur ↓ | Next-char acc ↑ |
|---|---:|---:|---:|---:|---:|
| raw | 3.7408 | 13.369 | 3.7343 | 3.7484 | 0.2893 |
| full | 3.3688 | 10.330 | 3.3816 | 3.3565 | 0.3284 |

13-gram overlap between eval set and training data (contamination check): raw 1.52%, +lang 1.52%, +heuristics 1.52%, +dedup 1.51%, full 1.48%
