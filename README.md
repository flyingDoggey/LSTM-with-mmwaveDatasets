# LSTM-with-mmwaveDatasets of gesture
mmwave Dataset by csv，and use LSTM to train it
## 这里是文件结构和说明
### 数据集
* 数据集完整文件在“手势识别毫米波.zip”文件中
* 数据集格式为csv格式，包含12个手势
* HMC.zip是我自己划分的训练集，HMC_TEST.zip是我划分的训练集
* 可根据自己需求更改
### 代码
* read_gestures.py为读取数据集文件方法，但在实际操作中不需要这么麻烦，可以忽略
* main.py 为使用lstm的主文件，使用lstm对数据集进行了训练
