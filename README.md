# TaurusCV

TaurusCV是一个基于深度学习的计算机视觉库，用于相关领域的学习和研究，旨在让大家对模型、数据和代码的复用更方便，降低深度学习框架学习的门槛。

现在开源的一些项目都太复杂了，本库希望能让大家用最快的速度理解一些cv架构的思路并快速实现。

所以，框架Taurus也会使用让人更容易理解和编码的方向实现。

### 项目结构

datasets 数据集获取

experiments 所有的运行和配置文件在这

layers 基础网络层

models 提供的可用模型

pretrained_models 公共预训练模型

utils 实用工具

visualization 可视化

### 使用方法

比如Faster rcnn，首先在experiments目录下创建自己的实验，然后配置好config.ini，里面包括了一些实验所需参数，预训练模型下载好放到pretrained_models目录中，然后执行实验里面的train.py即可。

### 更新日志

v1.0.0 初始版本，加入配置文件

