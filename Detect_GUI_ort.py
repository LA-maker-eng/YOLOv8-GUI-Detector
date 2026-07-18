import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import onnxruntime as ort
import ui_img.detect_images_rc

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.onnx'):
        self.model_path = model_path
        self.session = None
        self.input_shape = (640, 640)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def load_model(self):
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        
        pad_w = input_w - new_w
        pad_h = input_h - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        blob = cv2.dnn.blobFromImage(image_padded, 1/255.0, (input_w, input_h), swapRB=True, crop=False)
        
        return blob, scale, top, left, h, w
    
    def postprocess(self, outputs, scale, top, left, original_h, original_w, conf_threshold=0.5, iou_threshold=0.45):
        outputs = outputs[0]
        if outputs.shape[0] < outputs.shape[1]:
            outputs = outputs.transpose()
        
        rows = outputs.shape[0]
        
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(rows):
            conf = float(outputs[i, 4])
            if conf < conf_threshold:
                continue
            
            class_scores = outputs[i, 5:]
            class_scores = 1 / (1 + np.exp(-class_scores))
            class_id = int(np.argmax(class_scores))
            
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            
            class_score = float(class_scores[class_id])
            final_score = conf * class_score
            
            if final_score < conf_threshold:
                continue
            
            x_center, y_center, w, h = outputs[i, :4]
            x1 = int((x_center - w / 2 - left) / scale)
            y1 = int((y_center - h / 2 - top) / scale)
            x2 = int((x_center + w / 2 - left) / scale)
            y2 = int((y_center + h / 2 - top) / scale)
            
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(final_score)
            class_ids.append(class_id)
        
        if len(boxes) == 0:
            return []
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        
        results = []
        if indices is not None and len(indices) > 0:
            indices_flat = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in indices_flat:
                idx = int(i)
                if 0 <= idx < len(boxes):
                    results.append({
                        'box': boxes[idx],
                        'score': scores[idx],
                        'class_id': class_ids[idx],
                        'class_name': self.class_names[class_ids[idx]]
                    })
        
        return results
    
    def detect(self, image, conf_threshold=0.5):
        blob, scale, top, left, h, w = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: blob})
        results = self.postprocess(outputs, scale, top, left, h, w, conf_threshold)
        return results

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle("基于YOLOv8的检测演示软件V1.0")
        self.resize(1500, 1000)
        self.setStyleSheet("QWidget#centralwidget{background-image: url(:/detect_background/detect.JPG);}")
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")
        
        self.detector = YOLOv8Detector()
        self.detector.load_model()
        
        self.btn_selet_model = QtWidgets.QPushButton(self.centralwidget)
        self.btn_selet_model.setGeometry(QtCore.QRect(70, 810, 70, 70))
        self.btn_selet_model.setStyleSheet("border-image: url(:/detect_button_background/upload.png);")
        self.btn_selet_model.setText("")
        self.btn_selet_model.setObjectName("btn_selet_model")
        self.btn_selet_model.clicked.connect(self.seletModels)
        
        self.btn_detect_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_detect_img.setGeometry(QtCore.QRect(390, 810, 70, 70))
        self.btn_detect_img.setStyleSheet("border-image: url(:/detect_button_background/images.png);")
        self.btn_detect_img.setText("")
        self.btn_detect_img.setObjectName("btn_detect_img")
        self.btn_detect_img.clicked.connect(self.openImage)
        
        self.btn_save_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_img.setGeometry(QtCore.QRect(730, 810, 70, 70))
        self.btn_save_img.setStyleSheet("border-image: url(:/detect_button_background/save.png);")
        self.btn_save_img.setText("")
        self.btn_save_img.setObjectName("btn_save_img")
        self.btn_save_img.clicked.connect(self.saveImage)
        
        self.btn_clear_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear_img.setGeometry(QtCore.QRect(1050, 810, 70, 70))
        self.btn_clear_img.setStyleSheet("border-image: url(:/detect_button_background/delete.png);")
        self.btn_clear_img.setText("")
        self.btn_clear_img.setObjectName("btn_clear_img")
        self.btn_clear_img.clicked.connect(self.clearImage)
        
        self.label_img_input = QtWidgets.QLabel(self.centralwidget)
        self.label_img_input.setGeometry(QtCore.QRect(50, 110, 640, 480))
        self.label_img_input.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.label_img_input.setText("")
        self.label_img_input.setObjectName("label_img_input")
        
        self.label_img_output = QtWidgets.QLabel(self.centralwidget)
        self.label_img_output.setGeometry(QtCore.QRect(750, 110, 640, 480))
        self.label_img_output.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.label_img_output.setText("")
        self.label_img_output.setObjectName("label_img_output")
        
        self.label_text_input = QtWidgets.QLabel(self.centralwidget)
        self.label_text_input.setGeometry(QtCore.QRect(50, 80, 640, 30))
        self.label_text_input.setStyleSheet("background-color: rgb(255, 255, 255);color: rgb(0, 0, 0);")
        self.label_text_input.setText("待检测图像")
        self.label_text_input.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_input.setObjectName("label_text_input")
        
        self.label_text_output = QtWidgets.QLabel(self.centralwidget)
        self.label_text_output.setGeometry(QtCore.QRect(750, 80, 640, 30))
        self.label_text_output.setStyleSheet("background-color: rgb(255, 255, 255);color: rgb(0, 0, 0);")
        self.label_text_output.setText("检测结果")
        self.label_text_output.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_output.setObjectName("label_text_output")
        
        self.label_text_confidence = QtWidgets.QLabel(self.centralwidget)
        self.label_text_confidence.setGeometry(QtCore.QRect(50, 680, 100, 30))
        self.label_text_confidence.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_text_confidence.setText("置信度阈值")
        self.label_text_confidence.setObjectName("label_text_confidence")
        
        self.slider_confidence = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralwidget)
        self.slider_confidence.setGeometry(QtCore.QRect(150, 685, 500, 20))
        self.slider_confidence.setMinimum(10)
        self.slider_confidence.setMaximum(90)
        self.slider_confidence.setValue(50)
        self.slider_confidence.setObjectName("slider_confidence")
        
        self.label_confidence_value = QtWidgets.QLabel(self.centralwidget)
        self.label_confidence_value.setGeometry(QtCore.QRect(660, 680, 100, 30))
        self.label_confidence_value.setStyleSheet("background-color: rgb(255, 255, 255);color: rgb(0, 0, 0);")
        self.label_confidence_value.setText("当前值: 0.50")
        self.label_confidence_value.setObjectName("label_confidence_value")
        
        self.slider_confidence.valueChanged.connect(self.updateConfidence)
        
        self.text_browser = QtWidgets.QTextBrowser(self.centralwidget)
        self.text_browser.setGeometry(QtCore.QRect(50, 720, 1340, 70))
        self.text_browser.setStyleSheet("background-color: rgb(255, 255, 255);color: rgb(0, 0, 0);")
        self.text_browser.setText("检测信息将显示在这里")
        self.text_browser.setObjectName("text_browser")
        
        self.setCentralWidget(self.centralwidget)
        
        self.img_path = ""
        self.result_img = None
        
        QtCore.QMetaObject.connectSlotsByName(self)
    
    def updateConfidence(self):
        value = self.slider_confidence.value() / 100.0
        self.label_confidence_value.setText("当前值: {:.2f}".format(value))
    
    def seletModels(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "ONNX模型文件 (*.onnx)")
        if model_path:
            self.detector = YOLOv8Detector(model_path)
            self.detector.load_model()
            self.text_browser.setText("模型加载成功: {}".format(os.path.basename(model_path)))
    
    def openImage(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.jpg *.png *.jpeg)")
        if img_path:
            self.img_path = img_path
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 480))
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_show = QtGui.QImage(img_show.data, img_show.shape[1], img_show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_img_input.setPixmap(QtGui.QPixmap.fromImage(img_show))
            
            self.text_browser.setText("正在检测中...")
            QApplication.processEvents()
            
            conf_threshold = self.slider_confidence.value() / 100.0
            results = self.detector.detect(cv2.imread(img_path), conf_threshold)
            
            if len(results) > 0:
                self.text_browser.clear()
                self.text_browser.append("检测到 {} 个目标:".format(len(results)))
                
                result_img = cv2.imread(img_path)
                for i, result in enumerate(results):
                    box = result['box']
                    x1, y1, x2, y2 = box
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(result_img, "{}: {:.2f}".format(result['class_name'], result['score']),
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    self.text_browser.append("目标{}: {} (置信度: {:.2f})".format(i+1, result['class_name'], result['score']))
                
                result_img = cv2.resize(result_img, (640, 480))
                self.result_img = result_img
                img_show = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                img_show = QtGui.QImage(img_show.data, img_show.shape[1], img_show.shape[0], QtGui.QImage.Format_RGB888)
                self.label_img_output.setPixmap(QtGui.QPixmap.fromImage(img_show))
            else:
                self.text_browser.setText("未检测到目标")
                self.label_img_output.clear()
    
    def saveImage(self):
        if self.result_img is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", "图像文件 (*.jpg *.png)")
            if save_path:
                cv2.imwrite(save_path, self.result_img)
                self.text_browser.setText("检测结果已保存至: {}".format(save_path))
        else:
            self.text_browser.setText("没有可保存的检测结果")
    
    def clearImage(self):
        self.label_img_input.clear()
        self.label_img_output.clear()
        self.text_browser.setText("检测信息将显示在这里")
        self.img_path = ""
        self.result_img = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Ui_MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
