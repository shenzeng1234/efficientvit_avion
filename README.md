# Adapting EfficientViT to Video Action Recognition
EfficientViT is a new family of vision transformer models for efficient high resolution dense prediction (Cai et al, 2023). Its core is a new multi-scale linear attention module that enables global receptive field and multi-scale learning. The hallmark achievement of EfficientViT is its high efficiency that allows its deployment on diverse, resource-limited hardware platforms (e.g., mobile phones, network edge devices) while maintaining SOTA accuracy.

## Two-phased adaptation
Adpatation fo EfficientViT to video processing was carried out in two phases. 

#### Phase 1: EfficientViT on individual frames

Individual frames (images) were extracted from Epic-Kitechen 100 videos. Datasets that were coposed of extracted frame files were arranged required by PyTorch's DataFolder. Several EfficientViT models were trained on the frame datasets. Training accuracies were similar to Cai Han's results (in the mid-80's). Prediction accuracy on images was in the low 20%. Inference was performed on AWS 5.8xlarge instance and on Nvidia Jetson Orin Nano Developer Kit. Inference speed was in the milliseconds range.

#### Phase 2: Adapting EfficientViT with 3D convolutions

First, EfficientViT models are revised to accomodate video processing by incorporating 3D convolutions (substituting existing 2D convolutions) to model the nonlinear spatial-temporal relation between neighboring pixels. 3D inflation (Carreira, et al, 2017) was adopted to enable spatio-temporal feature extractions from video, which allows weights learned from training on ImageNet to be reused in training on videos. Additionally, the ReLU based linear self-attention was expanded from 2D in original EfficientViT to 3D. 

The revised models were trained on a resized Epic-Kitechen 100 video dataset. The framework in AVION (Zhao and Krahenbuhl, 2023) was updated for driving the training process.  

## Main observations

1. Using individual frames, EfficientViT training accuracy quickly saturated at ~22% of top 1% verb accuracy, and class prediction accuracy reached ~15%. Per-image inference time on both AWS 5.8xlarge and on Nvidia Jetson Orin Nano was in the milliseconds range.

2. Training of Efficient ViT model b1-r288 completed when the number of epochs reached the pre-specified 100 (1000 iterations per epoch). Top 1% verb accuracy reached 23.7%. When b1-r288 was pretrained with ImageNet initialization, Top 1% verb accuracy reached 27.7%, slightly higher than without pretraining.

## Next steps
1. With the current adaptation of the EfficientViT for video processing, training has been successful only for the b1-r288 model. Training of other EfficientViT models failed. Need to debug the network architecture so that training of other EfficientViT models could complete at pre-specified epoch numbers (e.g., 100) 
2. The training accuracy attained in the current EfficientViT adaptation is low. Next step is to fine tune the network architecture and hyperparamteters to increase training accuracy to near SOTA (60%).

I wrote a short report on this research. The report is available upon request.

## References
1. H. Cai, J. Li, M. Hu, C. Gan, and S. Han. Efficientvit: Multi-scale linear attention for high resolution dense prediction, 2023. URL https://doi.org/10.48550/arXiv.2205.14756. https://github.com/user-attachments/assets/ba56ead8-cada-4e84-9d82-1a4a1c69e7c6
2. J. Carreira and A. Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017. URL (https://github.com/user-attachments/assets/c049389f-272f-4155-a0a9-1cf8924adb82)
3. Y. Zhao and P. Krahenbuhl. Training a large video model on a single machine in a day, 2023. URL https://arxiv.org/pdf/2309.16669. URL (https://github.com/user-attachments/assets/6dc99946-2dd1-44d2-9924-e371b7370bf1)

**Manuscript accepted to publish in 2024 IEEE BigData Conference Proceedings**

