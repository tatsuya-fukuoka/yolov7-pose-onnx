# yolov7-pose-onnx
Infer yolov7-pose for images and videos with onnx model.

## 1.pip install
```bash
pip install -U pip && pip install opencv-python onnxruntime
```
## 2.pretraind-model
onnx model: yolov7-w6-pose-nms.onnx 

Taken from the official Github repository:
[trancongman276/yolov7-pose](https://github.com/trancongman276/yolov7-pose)

## 3.Inference
### 3.1 Image
```bash
python onnx_inference.py --mode image --model yolov7-w6-pose-nms.onnx -i sample.jpg -s 0.3 --img_size 960
```
### 3.2Video
```bash
python onnx_inference.py --mode video --model yolov7-w6-pose-nms.onnx -i sample.mp4 -s 0.3 --img_size 960
```

## 4.Reference
* [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/tree/pose)
* [trancongman276/yolov7-pose](https://github.com/trancongman276/yolov7-pose)
* [the exported onnx for yolo_pose seems not right](https://github.com/WongKinYiu/yolov7/issues/386)
