# RDSC
## How to use
* Replace RDSC directly with torch.nn.Conv2d.
* Note: Before loading the model parameters, it is necessary to conduct a few rounds of training in advance, in order to expand the deep convolution into a multi-form convolution matrix.
## 使用说明
* 将RDSC直接替换torch.nn.Conv2D即可
* 需要注意：模型加载参数之前需要提前进行几轮训练，才能将深度卷积拓展成多核卷积，才能成功加载模型参数
