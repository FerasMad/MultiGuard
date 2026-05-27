# Forensic Image Detector - Approach 1 (RGB + Fourier Mask) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\rgb_bandall\forensic_rgb_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9892 | 0.9410 | 0.9900 | 1000 |
| sdv1_4 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| sdv1_5 | Diffusion | _skipped_ | _skipped_ | _skipped_ | 0 |
| wukong | Diffusion | 0.9987 | 0.9820 | 0.9987 | 1000 |
| vqdm | Diffusion | 0.9979 | 0.9790 | 0.9980 | 1000 |
| adm | Diffusion | 0.9990 | 0.9870 | 0.9991 | 1000 |
| glide | Diffusion | 0.9994 | 0.9930 | 0.9995 | 1000 |
| biggan | GAN | 0.9998 | 0.9890 | 0.9998 | 1000 |
| **Overall Avg** | (all) | 0.9974 | 0.9785 | 0.9975 | - |
| **GAN Avg** | GAN | 0.9998 | 0.9890 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9969 | 0.9764 | 0.9970 | - |
| **Std Dev (AP)** | (across gens) | 0.0040 | - | - | - |

Generators evaluated: 6 / 8
