# 骨盆位置偵測專案

這個專案專注於骨盆位置的偵測和判斷。我們使用了mmdetection和mmpose的預訓練模型來實現這一目標。

## 前提條件

在開始之前，確保您已經安裝了以下框架：

- [mmpose](https://mmpose.readthedocs.io/zh_CN/latest/installation.html)
- [mmdetection](https://mmdetection.readthedocs.io/zh_CN/v2.25.0/get_started.html#id2)

## 安裝與使用

```bash
# 克隆此專案
git clone [您的專案連結]

# 安裝DVC
pip install dvc

# 設定DVC遠程儲存庫
dvc remote add -d storage gdrive://1Y9hJrHAqAoGMc4NunYC8ARZFH855H1WS

# 下載模型
dvc pull

# 推理您的照片
# 注意: 請先修改`inference.py`中的圖片路徑
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