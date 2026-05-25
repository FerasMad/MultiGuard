# Forensic Image Detector - Approach 1 (RGB+Fourier) + Approach 2 (DCT) - Per-Generator Eval

Per-generator AP / Accuracy / AUC for both approaches (F.25 + F.23).

| Generator | Type | A1 AP | A2 AP | A1 Acc | A2 Acc | A1 AUC | A2 AUC | n |
|-----------|------|-------|-------|--------|--------|--------|--------|---|
| midjourney | Diffusion | 0.9932 | 0.9569 | 0.9560 | 0.8590 | 0.9935 | 0.9606 | 1000 |
| sdv1_4 | Diffusion | _skipped_ | _skipped_ | _skipped_ | _skipped_ | _skipped_ | _skipped_ | 0 |
| sdv1_5 | Diffusion | _skipped_ | _skipped_ | _skipped_ | _skipped_ | _skipped_ | _skipped_ | 0 |
| wukong | Diffusion | 0.9992 | 0.9842 | 0.9830 | 0.9310 | 0.9992 | 0.9844 | 1000 |
| vqdm | Diffusion | 0.9980 | 0.9846 | 0.9750 | 0.9420 | 0.9980 | 0.9870 | 1000 |
| adm | Diffusion | 0.9987 | 0.9967 | 0.9830 | 0.9740 | 0.9987 | 0.9967 | 1000 |
| glide | Diffusion | 0.9986 | 0.9958 | 0.9900 | 0.9740 | 0.9988 | 0.9966 | 1000 |
| biggan | GAN | 0.9996 | 0.9998 | 0.9930 | 0.9830 | 0.9996 | 0.9998 | 1000 |
| **Overall Avg** | (all) | 0.9979 | 0.9863 | 0.9800 | 0.9438 | 0.9980 | 0.9875 | - |
| **GAN Avg** | GAN | 0.9996 | 0.9998 | 0.9930 | 0.9830 | 0.9996 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9975 | 0.9836 | 0.9774 | 0.9360 | 0.9976 | 0.9851 | - |
| **Std Dev (AP)** | (across gens) | 0.0023 | 0.0158 | - | - | - | - | - |

_Approach 1 (RGB+Fourier) evaluated 6/8 generators; Approach 2 (DCT) evaluated 6/8._

_Approach 1 ckpt: `phases\forensic\outputs\rgb\forensic_rgb_model.pth`_
_Approach 2 ckpt: `phases\forensic\outputs\dct\forensic_dct_model.pth`_
