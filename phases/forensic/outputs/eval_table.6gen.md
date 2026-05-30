# Forensic Image Detector - Approach 2 (DCT) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\dct\forensic_dct_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9569 | 0.8590 | 0.9606 | 1000 |
| sdv1_4 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| sdv1_5 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| wukong | Diffusion | 0.9842 | 0.9310 | 0.9844 | 1000 |
| vqdm | Diffusion | 0.9846 | 0.9420 | 0.9870 | 1000 |
| adm | Diffusion | 0.9967 | 0.9740 | 0.9967 | 1000 |
| glide | Diffusion | 0.9958 | 0.9740 | 0.9966 | 1000 |
| biggan | GAN | 0.9998 | 0.9830 | 0.9998 | 1000 |
| **Overall Avg** | (all) | 0.9863 | 0.9438 | 0.9875 | - |
| **GAN Avg** | GAN | 0.9998 | 0.9830 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9836 | 0.9360 | 0.9851 | - |
| **Std Dev (AP)** | (across gens) | 0.0158 | - | - | - |

Generators evaluated: 6 / 8
