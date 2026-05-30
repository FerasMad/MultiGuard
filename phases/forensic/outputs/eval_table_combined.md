# Forensic Image Detector - Approach 1 (RGB+Fourier) + Approach 2 (DCT) - Per-Generator Eval

Per-generator AP / Accuracy / AUC for both approaches (F.25 + F.23).

| Generator | Type | A1 AP | A2 AP | A1 Acc | A2 Acc | A1 AUC | A2 AUC | n |
|-----------|------|-------|-------|--------|--------|--------|--------|---|
| midjourney | Diffusion | 0.9759 | 0.9158 | 0.8510 | 0.7770 | 0.9787 | 0.9299 | 1000 |
| sdv1_4 | Diffusion | 0.9661 | 0.8381 | 0.9130 | 0.7820 | 0.9676 | 0.8582 | 1000 |
| sdv1_5 | Diffusion | 0.9670 | 0.8631 | 0.9060 | 0.7670 | 0.9690 | 0.8640 | 1000 |
| wukong | Diffusion | 0.9981 | 0.9828 | 0.9690 | 0.9010 | 0.9982 | 0.9816 | 1000 |
| vqdm | Diffusion | 0.9964 | 0.9859 | 0.9670 | 0.9310 | 0.9965 | 0.9866 | 1000 |
| adm | Diffusion | 0.9987 | 0.9952 | 0.9890 | 0.9630 | 0.9987 | 0.9952 | 1000 |
| glide | Diffusion | 0.9989 | 0.9938 | 0.9830 | 0.9770 | 0.9988 | 0.9959 | 1000 |
| biggan | GAN | 0.9995 | 0.9998 | 0.9970 | 0.9900 | 0.9996 | 0.9998 | 1000 |
| **Overall Avg** | (all) | 0.9876 | 0.9468 | 0.9469 | 0.8860 | 0.9884 | 0.9514 | - |
| **GAN Avg** | GAN | 0.9995 | 0.9998 | 0.9970 | 0.9900 | 0.9996 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9859 | 0.9393 | 0.9397 | 0.8711 | 0.9868 | 0.9445 | - |
| **Std Dev (AP)** | (across gens) | 0.0151 | 0.0654 | - | - | - | - | - |

_Approach 1 (RGB+Fourier) evaluated 8/8 generators; Approach 2 (DCT) evaluated 8/8._

_Approach 1 ckpt: `phases\forensic\outputs\rgb_8gen\forensic_rgb_model.pth`_
_Approach 2 ckpt: `phases\forensic\outputs\dct_8gen\forensic_dct_model.pth`_
