# Forensic Image Detector - Approach 1 (RGB + Fourier Mask) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\rgb_8gen\forensic_rgb_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9759 | 0.8510 | 0.9787 | 1000 |
| sdv1_4 | Diffusion | 0.9661 | 0.9130 | 0.9676 | 1000 |
| sdv1_5 | Diffusion | 0.9670 | 0.9060 | 0.9690 | 1000 |
| wukong | Diffusion | 0.9981 | 0.9690 | 0.9982 | 1000 |
| vqdm | Diffusion | 0.9964 | 0.9670 | 0.9965 | 1000 |
| adm | Diffusion | 0.9987 | 0.9890 | 0.9987 | 1000 |
| glide | Diffusion | 0.9989 | 0.9830 | 0.9988 | 1000 |
| biggan | GAN | 0.9995 | 0.9970 | 0.9996 | 1000 |
| **Overall Avg** | (all) | 0.9876 | 0.9469 | 0.9884 | - |
| **GAN Avg** | GAN | 0.9995 | 0.9970 | 0.9996 | - |
| **Diffusion Avg** | Diffusion | 0.9859 | 0.9397 | 0.9868 | - |
| **Std Dev (AP)** | (across gens) | 0.0151 | - | - | - |

Generators evaluated: 8 / 8
