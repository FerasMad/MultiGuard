# Forensic Image Detector - Approach 2 (DCT) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\dct_official\forensic_dct_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.8277 | 0.7370 | 0.8339 | 1000 |
| sdv1_4 | Diffusion | 0.8904 | 0.8090 | 0.8934 | 1000 |
| sdv1_5 | Diffusion | 0.8956 | 0.8170 | 0.9006 | 1000 |
| wukong | Diffusion | 0.9023 | 0.8180 | 0.9023 | 1000 |
| vqdm | Diffusion | 0.8302 | 0.7530 | 0.8478 | 1000 |
| adm | Diffusion | 0.9706 | 0.9090 | 0.9730 | 1000 |
| glide | Diffusion | 0.9750 | 0.9180 | 0.9753 | 1000 |
| biggan | GAN | 0.9761 | 0.9080 | 0.9796 | 1000 |
| **Overall Avg** | (all) | 0.9085 | 0.8336 | 0.9132 | - |
| **GAN Avg** | GAN | 0.9761 | 0.9080 | 0.9796 | - |
| **Diffusion Avg** | Diffusion | 0.8988 | 0.8230 | 0.9037 | - |
| **Std Dev (AP)** | (across gens) | 0.0610 | - | - | - |

Generators evaluated: 8 / 8
