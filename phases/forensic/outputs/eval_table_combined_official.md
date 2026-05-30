# Forensic Image Detector - Approach 1 (RGB+Fourier) + Approach 2 (DCT) - Per-Generator Eval

Per-generator AP / Accuracy / AUC for both approaches (F.25 + F.23).

| Generator | Type | A1 AP | A2 AP | A1 Acc | A2 Acc | A1 AUC | A2 AUC | n |
|-----------|------|-------|-------|--------|--------|--------|--------|---|
| midjourney | Diffusion | 0.9646 | 0.8277 | 0.9010 | 0.7370 | 0.9666 | 0.8339 | 1000 |
| sdv1_4 | Diffusion | 0.9879 | 0.8904 | 0.9520 | 0.8090 | 0.9868 | 0.8934 | 1000 |
| sdv1_5 | Diffusion | 0.9849 | 0.8956 | 0.9330 | 0.8170 | 0.9851 | 0.9006 | 1000 |
| wukong | Diffusion | 0.9795 | 0.9023 | 0.9280 | 0.8180 | 0.9810 | 0.9023 | 1000 |
| vqdm | Diffusion | 0.9764 | 0.8302 | 0.9200 | 0.7530 | 0.9776 | 0.8478 | 1000 |
| adm | Diffusion | 0.9974 | 0.9706 | 0.9690 | 0.9090 | 0.9973 | 0.9730 | 1000 |
| glide | Diffusion | 0.9872 | 0.9750 | 0.9540 | 0.9180 | 0.9878 | 0.9753 | 1000 |
| biggan | GAN | 0.9952 | 0.9761 | 0.9590 | 0.9080 | 0.9954 | 0.9796 | 1000 |
| **Overall Avg** | (all) | 0.9841 | 0.9085 | 0.9395 | 0.8336 | 0.9847 | 0.9132 | - |
| **GAN Avg** | GAN | 0.9952 | 0.9761 | 0.9590 | 0.9080 | 0.9954 | 0.9796 | - |
| **Diffusion Avg** | Diffusion | 0.9826 | 0.8988 | 0.9367 | 0.8230 | 0.9832 | 0.9037 | - |
| **Std Dev (AP)** | (across gens) | 0.0106 | 0.0610 | - | - | - | - | - |

_Approach 1 (RGB+Fourier) evaluated 8/8 generators; Approach 2 (DCT) evaluated 8/8._

_Approach 1 ckpt: `phases\forensic\outputs\rgb_official\forensic_rgb_model.pth`_
_Approach 2 ckpt: `phases\forensic\outputs\dct_official\forensic_dct_model.pth`_
