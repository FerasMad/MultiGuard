# Forensic Image Detector - Approach 1 (RGB + Fourier Mask) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\rgb\forensic_rgb_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9932 | 0.9560 | 0.9935 | 1000 |
| sdv1_4 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| sdv1_5 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| wukong | Diffusion | 0.9992 | 0.9830 | 0.9992 | 1000 |
| vqdm | Diffusion | 0.9980 | 0.9750 | 0.9980 | 1000 |
| adm | Diffusion | 0.9987 | 0.9830 | 0.9987 | 1000 |
| glide | Diffusion | 0.9986 | 0.9900 | 0.9988 | 1000 |
| biggan | GAN | 0.9996 | 0.9930 | 0.9996 | 1000 |
| **Overall Avg** | (all) | 0.9979 | 0.9800 | 0.9980 | - |
| **GAN Avg** | GAN | 0.9996 | 0.9930 | 0.9996 | - |
| **Diffusion Avg** | Diffusion | 0.9975 | 0.9774 | 0.9976 | - |
| **Std Dev (AP)** | (across gens) | 0.0023 | - | - | - |

Generators evaluated: 6 / 8
