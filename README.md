## Dynamic SRM Curriculum for Trustworthy Multi-modal Classification
> **Authors:**
Jian Zhu, Cui Yu, Xin Zou, Zhangmin Huang, Chenshu Hu, Jun Sun, Bo Lyu, Lei Liu, Chang Tang, Li-Rong Dai. 

This repo contains the code and data of our ICASSP'2025 paper [Dynamic SRM Curriculum for Trustworthy Multi-modal Classification](https://ieeexplore.ieee.org/abstract/document/10888724).

## 1. Abstract

Trustworthy multi-modal learning integrates multiple sources of data reliably. However, the current methods still focus on performance improvement by developing deep multi-modal networks. These approaches frequently encounter challenges due to the inherent non-convex nature of deep neural networks and their vulnerability to local minima, ultimately leading to a diminished ability for generalization. To address this problem, we present a novel curriculum termed the Dynamic SRM Curriculum (DSRMC). Within DSRMC, the deep trustworthy multi-modal networks undergo training with data provided sequentially, progressing from simple to complex samples. This training strategy mimics the human learning process, commencing with fundamental concepts and gradually advancing to tackle more complex and abstract ideas. Building upon DSRMC, we propose an innovative Curriculum Trustworthy Multi-modal Learning (CTML) method. CTML makes it easier to place the learned model in a flatter area, which improves its overall ability for generalization. Comprehensive experiments on three public datasets demonstrate that the proposed CTML performs better than state-of-the-art methods, achieving a maximum improvement of 6.7% on macroF1.

## 2.Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## 3.Datasets



## 4.Usage

- an example for train a new model：

```bash
python main.py
```



## 5.Citation

If you find our work useful in your research, please consider citing:

Zhu J, Yu C, Zou X, et al. Dynamic SRM Curriculum for Trustworthy Multi-modal Classification[C]//ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025: 1-5.

If you have any problems, contact me via qijian.zhu@outlook.com.


