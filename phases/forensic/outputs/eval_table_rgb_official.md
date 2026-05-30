# Forensic Image Detector - Approach 1 (RGB + Fourier Mask) - Per-Generator Eval

Checkpoint: `phases\forensic\outputs\rgb_official\forensic_rgb_model.pth`

| Generator | Type | AP | Accuracy | AUC | n_samples |
|-----------|------|----|----------|-----|-----------|
| midjourney | Diffusion | 0.9646 | 0.9010 | 0.9666 | 1000 |
| sdv1_4 | Diffusion | 0.9879 | 0.9520 | 0.9868 | 1000 |
| sdv1_5 | Diffusion | 0.9849 | 0.9330 | 0.9851 | 1000 |
| wukong | Diffusion | 0.9795 | 0.9280 | 0.9810 | 1000 |
| vqdm | Diffusion | 0.9764 | 0.9200 | 0.9776 | 1000 |
| adm | Diffusion | 0.9974 | 0.9690 | 0.9973 | 1000 |
| glide | Diffusion | 0.9872 | 0.9540 | 0.9878 | 1000 |
| biggan | GAN | 0.9952 | 0.9590 | 0.9954 | 1000 |
| **Overall Avg** | (all) | 0.9841 | 0.9395 | 0.9847 | - |
| **GAN Avg** | GAN | 0.9952 | 0.9590 | 0.9954 | - |
| **Diffusion Avg** | Diffusion | 0.9826 | 0.9367 | 0.9832 | - |
| **Std Dev (AP)** | (across gens) | 0.0106 | - | - | - |

Generators evaluated: 8 / 8
