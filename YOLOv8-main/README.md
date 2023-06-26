##  基于ultralytics YOLOv8 训练自己的数据集并基于NVIDIA TensorRT和openVINO进行推理

说明： 本项目支持YOLOv8的对应的package的版本是：[ultralytics-8.0.0](https://pypi.org/project/ultralytics/8.0.0/)

+ 模型配置文件：

```yaml
#myv8s.yaml
# Parameters
nc: 7  # number of classes
depth_multiple: 0.33  # scales module repeats
width_multiple: 0.50  # scales convolution channels

# YOLOv8.0s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0s head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)

```

+ 数据集配置文件

```yaml
#mydata.yaml
path: E:\\DataSets\\keypoint  # dataset root dir
train: images\\trainImages  # train images (relative to 'path') 4 images
val: images\\valImages  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes
names:
  0: rubberStopper
  1: pushRodTail
  2: needleTail
  3: mouth
  4: crookedMouth
  5: screwMouth
  6: smallRubberPlug

```

+ 训练超参数配置文件

我们对训练的超参数进行了简单的修改，通过命令行参数传入，也可以通过配置文件进行配置。
可以不修改下表的参数，而通过参数传入修改即可。

```yaml
task: "detect" # choices=['detect', 'segment', 'classify', 'init'] # init is a special case. Specify task to run.
mode: "train" # choices=['train', 'val', 'predict'] # mode to run task in.

# Train settings -------------------------------------------------------------------------------------------------------
model: null # i.e. yolov8n.pt, yolov8n.yaml. Path to model file
data: null # i.e. coco128.yaml. Path to data file
epochs: 100 # number of epochs to train for
patience: 50  # TODO: epochs to wait for no observable improvement for early stopping of training
batch: 5 # number of images per batch
imgsz: 640 # size of input images
save: True # save checkpoints
cache: False # True/ram, disk or False. Use cache for data loading
device: 0 # cuda device, i.e. 0 or 0,1,2,3 or cpu. Device to run on
workers: 6 # number of worker threads for data loading
project: null # project name
name: null # experiment name
exist_ok: False # whether to overwrite existing experiment
pretrained: False # whether to use a pretrained model
optimizer: 'SGD' # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
...
```

### YOLOv8目标检测任务训练

```shell 
 yolo task=detect mode=train model=yolov8s.yaml  data=mydata.yaml epochs=100 device=0 batch=5 workers=6 imgsz=640 pretrained=False optimizer=SGD 
```
```yaml 
main.py
if __name__ == '__main__':
    mode="onnx"
    if mode=="train":
        model=YOLO("yolov8s.yaml")
        # model=YOLO("yolov8s.pt")
        # model.train(**{'cfg':'D:\\AI\\YOLO\\yolov8-main\\ultralytics\\yolo\\cfg\\keypoints.yaml'})
        model.train(data='mydata.yaml',epochs=100,batch=5,device=0,workers=6)
        # path = model.export(format="onnx",opset=13)

    if mode=="onnx" :
        model = YOLO('runs\\detect\\train3\\weights\\best.pt') 
        path = model.export(format="onnx",opset=13)
    if mode=="predict" :
        model = YOLO('runs\\detect\\train3\\weights\\best.pt')
        # results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

        im1 = Image.open("E:\\DataSets\\keypoint\\images\\valImages\\20220706201141878.jpg")
        results = model.predict(source=im1, save=True)  # save plotted images

        # result=model("E:\\DataSets\\keypoint\\images\\valImages\\20220706201141878.jpg")
        # path = model.export(format='onnx')
        
        im2 = cv2.imread("E:\\DataSets\\keypoint\\images\\valImages\\20220719092357499.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
        # from list of PIL/ndarray
        results = model.predict(source=[im1, im2])

### YOLOv8推断Demo

```shell
# 自己实现的推断程序
python3 inference.py
```


### YOLOv8 TensorRT模型加速

1. pth模型转onnx

```shell
#CLI
yolo task=detect mode=export model=./runs/detect/train/weights/last.pt format=onnx simplify=True opset=13

# python 通过main.py实现
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt ")  # load a pretrained YOLOv8n model
model.export(format="onnx")  # export the model to ONNX format
```

2. 增加NMS Plugin 


执行`tensorrt/`下的如下代码，添加NMS到YOLOv8模型

+ 添加后处理

需要先修改onnx文件路径，以及classnums的类别数量。具体详见代码。
```shell
python3 yolov8_add_postprocess.py 
```

+ 添加NMS plugin
需要修改nms的参数。具体详见代码。
```shell
python3 yolov8_add_nms.py
```

生成`best_1_nms.onnx`,打开该文件对比和原onnx文件的区别，发现增加了如下节点(完成了将NMS添加到onnx的目的）：

使用netron打开onnx查看网络最后输出的格式。


3. onnx转trt engine

```shell
trtexec --onnx=best_1_nms.onnx --saveEngine=yolov8s.engine --workspace=3000 --verbose
```

4. TRT C++推断
运行yolov8_trt.cpp直接进行推理，结果保存在指定的文件夹中。

在win 10下基于RTX 2060 TensorRT 8.5.1进行测试，我们的开发环境是VS2019,**所有C++代码已经存放在`tensorrt/`文件夹下**。其推断结果如下图所示（可以发现我们实现了YOLOv8的TensorRT端到端的推断，其推断结果与原训练框架保持一致）：

PS：tensorrt需要根据不同的开发部署平台配置相应的环境变量。


### YOLOv8实现OpenVINO模型推断加速
---工具和环境安装：
参考官网：https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=WINDOWS&VERSION=v_2023_0&DISTRIBUTION=PIP

---模型使用：
1.使用best.onnx
   openvino支持直接加载onnx模型进行推理。
2.使用IR模型
  --1.首先激活openvino的环境变量：openvino_env\Scripts\activate
  --2. 模型转换：将训练得到的ONNX模型转换为XML和Bin的IR模型，输出为FP16格式。
    mo   --input_model D:\AI\YOLO\yolov8-data\YOLOv8-main\runs\detect\train\weights\best.onnx     --output_dir D:\AI\YOLO\yolov8-data\YOLOv8-main\runs\detect\train\weights\   --compress_to_fp16
  --3.转换完成后得到模型文件：xml和bin文件。
3.运行yolov8_openvino.cpp进行推理，结果保存在指定的文件夹中。

### 建议
   【建议使用最新的yolov8，然后进行预训练权重加载后训练，得到模型后在使用本项目提供的tensorrt或者openvino进行推理。当前版本yolov8较旧，很多新的功能没有，不建议使用，仅作展示，例如没有加载预训练模型权重的功能。】

### 参考文献：

+ https://github.com/ultralytics/ultralytics
+ https://github.com/DataXujing/YOLOv8
+ https://mp.weixin.qq.com/s/_OvSTQZlb5jKti0JnIy0tQ
+ https://github.com/ultralytics/assets/releases
+ https://v8docs.ultralytics.com/
+ https://pypi.org/project/ultralytics/0.0.44/#description
+ https://mp.weixin.qq.com/s/-4pn--3kFI_J1oX6p5GWVQ
+ https://github.com/uyolo1314/ultralytics



