# FERDenoise

Code for "ReSup: Reliable Label Noise Suppression for Facial Expression Recognition", which have been submitted to IEEE Transactions on Affective Computing.

## Abstract

Because of the ambiguous and subjective property of the facial expression, the label noise is widely existing in the FER dataset. For this problem, in the training phase, current methods often directly predict whether the label is noised or not, aiming to reduce the contribution of the noised data. However, we argue that this kind of method suffers from the low reliability of such noise data decision operation. It makes that some mistakenly abounded clean data are not utilized sufficiently and some mistakenly kept noised data disturbing the model learning. In this paper, we propose a more reliable noise-label suppression method called ReSup. First, instead of directly predicting noised or not, ReSup makes the noise data decision by modeling the distribution of noise and clean labels simultaneously according to the disagreement between the prediction and the target. Specifically, to achieve optimal distribution modeling, ReSup models the similarity distribution of all samples. To further enhance the reliability of our noise decision results, ReSup uses two networks to jointly achieve noise suppression. Specifically, ReSup utilize the property that two networks are less likely to make the same mistakes, making two networks swap decisions and tending to trust decisions with high agreement. Extensive experiments on popular datasets shows the effectiveness of ReSup.

## ReSup network 

![image](https://github.com/purpleleaves007/FERDenoise/assets/49738655/b45fa3b8-03c5-48c1-a0ab-d9560bb17b2b)

## Results

![image](https://github.com/purpleleaves007/FERDenoise/assets/49738655/7ce3f7e9-cf92-4822-a10d-a97c6f7db785)

![image](https://github.com/purpleleaves007/FERDenoise/assets/49738655/e518f6c4-f229-4c22-9ca2-8d5b2c72f552)

If it's useful, please cite
@article{zhang2023resup,
  title={ReSup: Reliable Label Noise Suppression for Facial Expression Recognition},
  author={Zhang, Xiang and Lu, Yan and Yan, Huan and Huang, Jingyang and Ji, Yusheng and Gu, Yu},
  journal={arXiv preprint arXiv:2305.17895},
  year={2023}
}
