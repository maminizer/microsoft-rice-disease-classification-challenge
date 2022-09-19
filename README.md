# Microsoft rice classification challenge
### Microsoft rice classification challenge [Competition](https://zindi.africa/competitions/microsoft-rice-disease-classification-challenge)
##### Top 15% place Solution [leaderboard](https://zindi.africa/competitions/microsoft-rice-disease-classification-challenge/leaderboard)
#### Model OverView :  [ArcFace Sub-center Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
***Abstract.*** Margin-based deep face recognition methods (e.g. SphereFace,
CosFace, and ArcFace) have achieved remarkable success in unconstrained
face recognition. However, these methods are susceptible to the ***massive label noise*** in the training data and thus require laborious human effort
to clean the datasets. In this paper, we relax the ***intra-class constraint
of ArcFace*** to ***improve the robustness to label noise.*** More specifically,
we design K sub-centers for each class and the training sample only
needs to be close to any of the K positive sub-centers instead of the
only one positive center. The proposed sub-center ArcFace encourages
one dominant sub-class that contains the majority of clean faces and
non-dominant sub-classes that include hard or noisy faces. Extensive
experiments confirm ***the robustness of sub-center ArcFace*** under massive real-world noise. After the model achieves enough discriminative
power, we directly drop non-dominant sub-centers and high-confident
noisy samples, which helps recapture intra-compactness, decrease the influence from noise, and achieve comparable performance compared to
ArcFace trained on the manually cleaned dataset. By taking advantage
of the large-scale raw web faces (Celeb500K), sub-center Arcface achieves
state-of-the-art performance on IJB-B, IJB-C, MegaFace, and FRVT.

## DATA SETUP : Assumes that you have [kaggleAPI](https://github.com/Kaggle/kaggle-api) installed
```
kaggle competitions download -c microsoft-rice-disease-classification-challenge
unzip -q microsoft-rice-disease-classification-challenge.zip
```

## Usage 
```
# Train Model:
python src/preprocess.py
python src/Train.py
```

### Acknowledgement
- Y.NAKAMA [notebook](https://www.kaggle.com/yasufuminakama/herbarium-2020-pytorch-resnet18-train/notebook)
- Haqishen [repository](https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution)
