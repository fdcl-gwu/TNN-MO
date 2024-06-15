# Deep Monocular Relative 6D Pose Estimation for Ship-Based Autonomous UAV

## Model Testing with Real Data
The TNN-MO model is tested with the real world images captured by the DCS over multiple days. 
During the in-flight experiments conducted on the USNA research vessel, the model was subjected to real-world conditions where instances of variable lighting and occlusions were encountered. 
We deliberately selected challenging images for testing because they represent realistic conditions where such situations can occur.

### Overexposed ship
An overexposed ship is one where the image has captured too much light, causing the ship to appear excessively bright.
This results in a loss of detail, especially in areas that  have subtle color variations or textures. 
The ship’s features become hard to be distinguished because the intense light overwhelms the camera’s sensor, leading to a predominance of white or light areas, particularly on surfaces like the landing pad. It’s as if the ship is caught in a glare, with its details bleached by the brightness.

<img src="results/Overexposed.gif" alt="color picker" />

### Underexposed ship
An underexposed ship is one where the image has not captured enough light, making the ship appear too dark. 
This can obscure details and make it challenging to distinguish features, especially in areas that are naturally shadowed or lack reflective surfaces.
It is as if the ship is enveloped in shadows, with its details concealed in darkness.

<img src="results/Underexposed.gif" alt="color picker" />

### Normal ship
A normal ship, in terms of exposure, is one where the lighting conditions are ideal, resulting in a balanced image with clear visibility of details. 
The lighting is neither too intense nor too dim, providing an optimal level of brightness that allows all parts of the ship to be distinguished clearly.

<img src="results/Normal.gif" alt="color picker" />

<!-- ##  Data
- **`31 Aug, 2023`:**   -->
## Validation with RTK GPS
For quantitative validation, the estimated pose is compared with the actual pose determined by the Data Collection System (DCS). The DCS integrates the relative attitude, derived from the base rover's IMUs, and the relative position from the RTK-GPS with an extended Kalman filter, where the RTK-GPS provides a centimeter-level accuracy in nominal conditions. Here, the measurements of the RTK-GPS are considered as ground truth.

The trajectories estimated by the proposed TNN-MO model and the RTK-GPS under the above three illumination conditions are illustrated in above with the estimation errors and it is shown that the position and the attitude trajectory estimated by the TNN-MO model are consistent with the IMU and the RTK-GPS.

### Accuracy of 6D Pose Estimation for TNN-MO Model Under Variable Lighting Conditions
| Image Type | Max Range, $L$ (m) | MAE / $\sigma$ / $d$ of Rot. (deg) | MAE / $\sigma$ of Pos. (m) | MAE/$L$ (\%) |
| --- | --- | --- | --- | --- |
| Overexposed Ship | 13.7 | 1.8 / 2.32 / 0.999 | 0.112 / 0.017 | 0.82 |
| Underexposed Ship | 13.5 | 1.1 / 1.83 / 0.999 | 0.089 / 0.019 | 0.66 |
| Normal Ship | 18.2 | 4.0 / 4.54 / 0.999 | 0.177 / 0.022 | 0.97 |


## Installation

Install packages using `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Dataset

## Training

```bash
sh train.sh
```

The TNN-MO model was trained for 350 epochs with the batch size of 48, hyperparameter $\gamma = 10$, leveraging the AdamW optimizer, using a 20GB Multi-Instance GPU (MIG) partition from NVIDIA A100-PCIE-40GB GPU.


## Checkpoints

```
📁 TNN-MO/
  ├── 📁 checkpoints/
    ├── 📁 TNN_MO_6-Object_model/
      ├── 📦 TNN_MO_6-Object_model.pth
  	  └── 📄 config.ini
  └── 📁 examples/
  	├── 📁 Test_real
  	└── 📁 Test_syn
```

## Acknowledgement
Our code is based on [DETR](https://github.com/facebookresearch/detr). Thanks for their wonderful works.

## Citation

If you use TNN-MO in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{wickramasuriya2024deep,
  title={Deep Transformer Network for Monocular Pose Estimation of Ship-Based UAV},
  author={Maneesha Wickramasuriya and Taeyoung Lee and Murray Snyder},
  journal={arXiv preprint arXiv:2406.09260},
  year={2024}
}
```
