import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'windows'
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QFileDialog,
                             QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.onnx'):
        self.model_path = model_path
        self.net = None
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
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            print('模型加载成功')
        except Exception as e:
            print(f'模型加载失败: {e}')
    
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
            class_id = int(np.argmax(class_scores))
            
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            
            final_score = conf
            
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
        if self.net is None:
            self.load_model()
        
        blob, scale, top, left, h, w = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward()
        results = self.postprocess(outputs, scale, top, left, h, w, conf_threshold)
        
        return image, results
    
    def draw_results(self, image, results):
        image_copy = image.copy()
        colors = {
            'person': (0, 0, 255), 'bicycle': (0, 255, 0), 'car': (255, 0, 0),
            'motorcycle': (0, 255, 255), 'airplane': (255, 0, 255), 'bus': (255, 255, 0),
            'train': (128, 0, 0), 'truck': (0, 128, 0), 'boat': (0, 0, 128),
            'traffic light': (128, 128, 0), 'fire hydrant': (128, 0, 128),
            'stop sign': (0, 128, 128), 'parking meter': (64, 64, 64),
            'tie': (0, 255, 255)
        }
        
        for result in results:
            x1, y1, x2, y2 = result['box']
            class_name = result['class_name']
            score = result['score']
            color = colors.get(class_name, (0, 0, 255))
            
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            label = f'{class_name}: {score:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_copy

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle('基于YOLOv8的检测演示软件V1.0')
        self.resize(1000, 700)
        
        self.detector = YOLOv8Detector()
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        top_layout = QHBoxLayout()
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.label_original = QLabel('原始图像')
        self.label_original.setFixedSize(400, 300)
        self.label_original.setStyleSheet('border: 2px solid gray; background-color: #f0f0f0;')
        self.label_original.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.label_original)
        
        left_layout.addStretch()
        top_layout.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.label_detect = QLabel('检测图像')
        self.label_detect.setFixedSize(400, 300)
        self.label_detect.setStyleSheet('border: 2px solid blue; background-color: #f0f0f0;')
        self.label_detect.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.label_detect)
        
        self.label_info = QLabel('检测信息：等待加载图像...')
        self.label_info.setStyleSheet('font-size: 12px; color: #333;')
        right_layout.addWidget(self.label_info)
        
        right_layout.addStretch()
        top_layout.addWidget(right_panel)
        
        top_layout.addStretch()
        main_layout.addLayout(top_layout)
        
        center_layout = QHBoxLayout()
        
        conf_group = QWidget()
        conf_layout = QVBoxLayout(conf_group)
        
        conf_label = QLabel('置信度阈值')
        conf_label.setFont(QFont('Arial', 10, QFont.Bold))
        conf_layout.addWidget(conf_label)
        
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(50)
        self.slider_conf.setTickInterval(10)
        self.slider_conf.setTickPosition(QSlider.TicksBelow)
        conf_layout.addWidget(self.slider_conf)
        
        self.label_conf_value = QLabel('当前值: 0.50')
        conf_layout.addWidget(self.label_conf_value)
        
        self.slider_conf.valueChanged.connect(self.on_conf_changed)
        
        center_layout.addWidget(conf_group)
        center_layout.addStretch()
        
        main_layout.addLayout(center_layout)
        
        bottom_layout = QHBoxLayout()
        
        self.btn_select_model = QPushButton('模型选择')
        self.btn_select_model.setFixedSize(100, 40)
        self.btn_select_model.clicked.connect(self.SelectModel)
        bottom_layout.addWidget(self.btn_select_model)
        
        self.btn_detect_img = QPushButton('图像加载')
        self.btn_detect_img.setFixedSize(100, 40)
        self.btn_detect_img.clicked.connect(self.openImage)
        bottom_layout.addWidget(self.btn_detect_img)
        
        self.btn_detect_cam = QPushButton('摄像头检测')
        self.btn_detect_cam.setFixedSize(120, 40)
        self.btn_detect_cam.clicked.connect(self.detectCamera)
        bottom_layout.addWidget(self.btn_detect_cam)
        
        self.btn_save = QPushButton('图像保存')
        self.btn_save.setFixedSize(100, 40)
        self.btn_save.clicked.connect(self.saveImage)
        bottom_layout.addWidget(self.btn_save)
        
        self.btn_clear = QPushButton('图像清除')
        self.btn_clear.setFixedSize(100, 40)
        self.btn_clear.clicked.connect(self.clearImage)
        bottom_layout.addWidget(self.btn_clear)
        
        self.btn_exit = QPushButton('应用退出')
        self.btn_exit.setFixedSize(100, 40)
        self.btn_exit.clicked.connect(self.close_app)
        bottom_layout.addWidget(self.btn_exit)
        
        bottom_layout.addStretch()
        main_layout.addLayout(bottom_layout)
        
        self.original_image = None
        self.detect_image = None
        self.cap = None
        self.timer = None
        
    def on_conf_changed(self, value):
        self.label_conf_value.setText(f'当前值: {value/100:.2f}')
        if self.original_image is not None and self.cap is None:
            self.redetect_image()
    
    def redetect_image(self):
        conf_threshold = self.slider_conf.value() / 100.0
        
        try:
            _, results = self.detector.detect(self.original_image, conf_threshold)
            self.detect_image = self.detector.draw_results(self.original_image, results)
            self.showImage(self.detect_image, self.label_detect)
            
            info_text = f'检测到 {len(results)} 个目标：'
            for result in results:
                info_text += f'\n- {result["class_name"]} (置信度: {result["score"]:.2f})'
            if len(results) == 0:
                info_text = '未检测到目标'
            self.label_info.setText(info_text)
            
        except Exception as e:
            self.label_info.setText(f'检测错误: {str(e)}')
    
    def SelectModel(self):
        QMessageBox.information(self, '提示', '当前使用YOLOv8n-nano模型 (ONNX Runtime模式)')
    
    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                QMessageBox.warning(self, '警告', '无法读取图片文件！')
                return
            
            self.showImage(self.original_image, self.label_original)
            self.redetect_image()
    
    def detectCamera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                QMessageBox.warning(self, '警告', '无法打开摄像头！')
                self.cap = None
                return
            
            self.btn_detect_cam.setText('停止检测')
            
            self.timer = QTimer()
            self.timer.timeout.connect(self.updateCamera)
            self.timer.start(30)
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.btn_detect_cam.setText('摄像头检测')
            self.label_original.clear()
            self.label_original.setText('原始图像')
            self.label_detect.clear()
            self.label_detect.setText('检测图像')
    
    def updateCamera(self):
        ret, frame = self.cap.read()
        if ret:
            self.showImage(frame, self.label_original)
            
            conf_threshold = self.slider_conf.value() / 100.0
            try:
                _, results = self.detector.detect(frame, conf_threshold)
                detect_frame = self.detector.draw_results(frame, results)
                self.showImage(detect_frame, self.label_detect)
                
                self.label_info.setText(f'检测到 {len(results)} 个目标')
            except Exception as e:
                self.label_info.setText(f'检测错误: {str(e)}')
    
    def showImage(self, cv_image, label):
        cv_image_resized = cv2.resize(cv_image, (400, 300))
        h, w, ch = cv_image_resized.shape
        bytes_per_line = ch * w
        qt_image = QImage(cv_image_resized.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def saveImage(self):
        if self.detect_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, '保存图片', '', 'Images (*.png *.jpg)')
            if file_path:
                cv2.imwrite(file_path, self.detect_image)
                QMessageBox.information(self, '提示', '图片保存成功！')
        else:
            QMessageBox.warning(self, '警告', '没有可保存的检测结果！')
    
    def clearImage(self):
        self.label_original.clear()
        self.label_original.setText('原始图像')
        self.label_detect.clear()
        self.label_detect.setText('检测图像')
        self.label_info.setText('检测信息：等待加载图像...')
        self.original_image = None
        self.detect_image = None
        
        if self.cap is not None:
            self.detectCamera()
    
    def close_app(self):
        reply = QMessageBox.question(self, '退出确认', '确定要退出应用吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.cap is not None:
                self.cap.release()
            self.close()
    
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())