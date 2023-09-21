# 骨盆位置侦测项目

这个专案专注于骨盆位置的侦测和判断。我们使用了mmdetection和mmpose的预训练模型来实现这一目标。

## 前提条件

在开始之前，确保您已经安装了以下框架：

- [mmpose](https://mmpose.readthedocs.io/zh_CN/latest/installation.html)
- [mmdetection](https://mmdetection.readthedocs.io/zh_CN/v2.25.0/get_started.html#id2)

## 安装与使用

```bash
# 克隆此专案
git clone [您的专案链接]

# 安装DVC
pip install dvc

# 设置DVC远程储存库
dvc remote add -d storage gdrive://1Y9hJrHAqAoGMc4NunYC8ARZFH855H1WS

# 下载模型
dvc pull

# 推理您的照片
# 注意: 请先修改`inference.py`中的图片路径
python inference.py
```
## 引文

### MMPose 引文
```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
### MMDetection 引文
```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```