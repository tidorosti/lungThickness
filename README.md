# U-Net-based Lung Thickness Map for Pixel-level Lung Volume Estimation of Chest X-rays

Code to the paper: "U-Net-based Lung Thickness Map for Pixel-level Lung Volume Estimation of Chest X-rays."

## Getting Started

### Dependencies:
- astra-toolbox (version 2.1.3)
- lungmask (version 0.2.20)
- xraylib (version 4.1.3)
- spekpy (version 2.0.1)

### Executing program

- Obtain public data:
   - Luna16: https://www.kaggle.com/datasets/avc0706/luna16 and see https://luna16.grand-challenge.org/
   - RSNA PE challenge: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data and see https://www.rsna.org/rsnai/ai-image-challenge/rsna-pe-detection-challenge-2020
- generate synthetic radiographs with 1_simulation.ipynb
- Train network with 2_training.ipynb
- Test model and obtain predicted thickness maps and total lung volume estimations with 3_evaluation.ipynb

------------------------
 Authored by:
- M. Schultheiß
- T. Dorosti

With contributions from:
- P. Schmette
- J. Heuchert
