# Forensic Image Detector - Approach 2 (DCT) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\dct_8gen\forensic_dct_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9158 | 0.7770 | 0.9299 | 1000 |
| sdv1_4 | Diffusion | 0.8381 | 0.7820 | 0.8582 | 1000 |
| sdv1_5 | Diffusion | 0.8631 | 0.7670 | 0.8640 | 1000 |
| wukong | Diffusion | 0.9828 | 0.9010 | 0.9816 | 1000 |
| vqdm | Diffusion | 0.9859 | 0.9310 | 0.9866 | 1000 |
| adm | Diffusion | 0.9952 | 0.9630 | 0.9952 | 1000 |
| glide | Diffusion | 0.9938 | 0.9770 | 0.9959 | 1000 |
| biggan | GAN | 0.9998 | 0.9900 | 0.9998 | 1000 |
| **Overall Avg** | (all) | 0.9468 | 0.8860 | 0.9514 | - |
| **GAN Avg** | GAN | 0.9998 | 0.9900 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9393 | 0.8711 | 0.9445 | - |
| **Std Dev (AP)** | (across gens) | 0.0654 | - | - | - |

Generators evaluated: 8 / 8
