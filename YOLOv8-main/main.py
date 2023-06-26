from ultralytics import YOLO
from PIL import Image
import cv2
import onnx
if __name__ == '__main__':
    mode="train"
    if mode=="train":
        # model=YOLO("myv8s.yaml") 
        model=YOLO("yolov8s.pt")
        # model.train(**{'cfg':'D:\\AI\\YOLO\\yolov8-main\\ultralytics\\yolo\\cfg\\keypoints.yaml'})
        model.train(data='mydata.yaml',epochs=100,batch=4,device=0,workers=6)
        # path = model.export(format="onnx",opset=13)

    if mode=="onnx" :
        model = YOLO('runs\\detect\\train4\\weights\\best.pt') 
        path = model.export(format="onnx",opset=13)
    if mode=="predict" :
        model = YOLO('runs\\detect\\train\\weights\\best.pt')
        # results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

        im1 = Image.open("E:\\DataSets\\keypoint\\images\\valImages\\20220706201141878.jpg")
        results = model.predict(source=im1, save=True)  # save plotted images

        # result=model("E:\\DataSets\\keypoint\\images\\valImages\\20220706201141878.jpg")
        # path = model.export(format='onnx')
        
        im2 = cv2.imread("E:\\DataSets\\keypoint\\images\\valImages\\20220719092357499.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
        # from list of PIL/ndarray
        results = model.predict(source=[im1, im2])