# SVT-SDE

SVT-SDE: Spatiotemporal Vision Transformers-based Selfsupervised Depth Estimation in Stereoscopic Surgical Videos

This is a reference implementation of our paper recently published on IEEE Transactions on Medical Robotics and Bionics

Rong Tao, Baoru Huang, Xiaoyang Zou, Guoyan Zheng* <br/>

Network design:
![alt text](https://github.com/gloriatao/SVT-SDE/blob/main/Fig3.png)


Loss design:
![alt text](https://github.com/gloriatao/SVT-SDE/blob/main/Fig2.png)


Performance on the SCARED dataset
![alt text](https://github.com/gloriatao/SVT-SDE/blob/main/Fig.SCARED_dataset8.png)

Dense depth estimation plays a crucial role in developing context-aware computer-assisted intervention systems. However, it is a challenging task due to low image quality and
highly dynamic surgical environment. This problem is furthercomplicated by the difficulty in acquiring per-pixel ground truth depth data in a surgical setting. Recent works on self-supervised
depth estimation use reconstructed images as supervisory signal, which helps to eliminate the requirement of ground truth data but also causes over-smoothed results on small structures and
boundary edges. Additionally, most surgical depth estimation methods are built upon static laparoscopic images, ignoring rich temporal information. To address these challenges, we
propose a novel spatiotemporal vision transformers-based selfsupervised depth estimation method, referred as SVT-SDE, for accurate and robust depth estimation from paired surgical videos.
Unlike previous works, our method leverages a novel design of spatiotemporal vision transformers (SVT) to learn the complementary information of visual and temporal features learned
from the stereoscopic video sequences.We further introduce highfrequency-based image reconstruction supervisory losses, which help to preserve fine-grained details of the surgical scene. Results
from experiments conducted on two publicly available datasets demonstrate the superior
