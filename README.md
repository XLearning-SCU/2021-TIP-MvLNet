# Deep Spectral Representation Learning From Multi-View Data

This repository is the official implementation of [MvLNet](https://ieeexplore.ieee.org/abstract/document/9446572). 

<img src="https://github.com/XLearning-SCU/2021-TIP-MvLNet/blob/main/MvLNet.png"  width="1024" height="331" />

## Requirements

Ubuntu 16.04 + cuda9.0

```setup
tensorflow-gpu==1.9.0
Keras==2.1.6
numpy==1.14.3
scikit-learn==0.19.1
munkres==1.0.12
```

## Pre-trained Models & Dataset

Download pre-trained siamese networks, autoencoders and datasets here:

- [MvLNet](https://drive.google.com/drive/folders/1mYEuh_VPcTA3Ff9YGv8hOHpfuRWPA8bU?usp=sharing)


## Training

To train and evaluate the model in the paper, run this command:

```train & evaluate
python run.py
```

>📋  The default training scrip trains MvLNet on Noisy MNIST. Replace the config name in run.py for other datasets, i.e. Caltech101-20 or wiki.

## Results

Our model achieves the following performance :

### Clustering on Noisy MNIST

| Model name         | ACC          |    F-mea   |   NMI    |   AMI   |
| ------------------ | ------------ | ---------- | -------- | ------- |
| MvLNet             |  99.18       |    99.16   |   97.76  |   97.75 |

### Classification on Caltech101-20

| Model name         | ACC             | F-mea          | Precision |
| ------------------ |---------------- | -------------- | --------- |
| MvLNet             |     84.49       |      83.57     |    84.29  |

### Retreival on Wikipedia

| Model name         | Image -> Text   | Text -> Image  |     AVG   |
| ------------------ |---------------- | -------------- | --------- |
| MvLNet             |     38.7        |      44.4      |    41.5   |

## Citation

If you find our work useful in your research, please consider citing:

```
@ARTICLE{huang2021deep,
  author={Huang, Zhenyu and Zhou, Joey Tianyi and Zhu, Hongyuan and Zhang, Changqing and Lv, Jiancheng and Peng, Xi},
  journal={IEEE Transactions on Image Processing}, 
  title={Deep Spectral Representation Learning From Multi-View Data}, 
  year={2021},
  volume={30},
  number={},
  pages={5352-5362},
  doi={10.1109/TIP.2021.3083072},
}
```