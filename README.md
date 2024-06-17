### 项目介绍

ResNet网络的应用—抑郁症诊断

使用数据集：**AVEC2014**

数据集下载地址 <a href="https://pan.baidu.com/s/15C75G_PDpwqRmdb754V6Bw?pwd=tmup">AVEC2014</a>

提取码：AVEC

预处理：

​	1.**采样**，AVEC2013每个视频取100帧，保留原始label

​	2.**人脸对齐裁剪**，使用**MTCNN**工具

### 文件介绍

```
preprocess.py	主要用于预处理视频信息，从中提取帧，并在视频帧中提取人脸
函数：generate_label_file()	将运来的label合并为一个csv文件
函数：get_img()	抽取视频帧，每个视频按间隔抽取100-105帧
函数：get_face()	使用MTCNN提取人脸，并分割图片

model.py	模型的网络结构
```

```
load_data.py	获取图片存放路径以及将标签与之对应
writer.py	创建Tensorboard记录器，保存训练过程损失
dataset.py	继承torch.utils.Dataset,负责将数据转化为torch.utils.data.DataLoader可以处理的迭代器
train.py	模型训练
validate.py	验证模型
test.py		测试模型的性能，并记录预测分数，保存在testInfo.csv,记录了每张图片的路径，label,预测分数
main.py		模型训练入口文件
```

```
img		提取的视频帧文件
log		Tensorboard日志文件
model_dict	训练好的模型参数文件
processed	存放预处理完成之后的人脸图片，label文件
AVEC2014	数据集存放位置
```

```
查看训练日志方法：
	安装tensorboard库之后，输入命令tensorboard --lofdir log_dir_path,打开命令执行后出现的网址即可
	log_dir_path是存放Tensorboard日志文件的文件夹路径
```

```
运行顺序:preprocess.py--->main.py--->test.py
```
