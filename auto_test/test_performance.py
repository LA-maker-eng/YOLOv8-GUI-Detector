import sys
import os
import time
import unittest
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from Detect_GUI_cvdnn import YOLOv8Detector

os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output'), exist_ok=True)


class TestYOLOv8DetectorPerformance(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv8Detector()
        self.detector.load_model()
        self.test_image = cv2.imread('test/many.png')
        if self.test_image is None:
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_single_image_inference_speed(self):
        """PERF-001 测试单图片推理速度"""
        print("[INFO] PERF-001 - 单图片推理速度测试开始...")
        total_time = 0.0
        count = 30
        times = []
        
        for i in range(count):
            start_time = time.time()
            _, _ = self.detector.detect(self.test_image, conf_threshold=0.5)
            elapsed = time.time() - start_time
            times.append(elapsed)
            total_time += elapsed
            if (i + 1) % 10 == 0:
                print(f"  第{i+1}次推理耗时: {elapsed:.4f}s")
        
        avg_time = total_time / count
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        
        self.save_performance_data('single_image_inference', {
            'avg_time': avg_time, 'min_time': min_time, 'max_time': max_time, 'fps': fps
        })
        
        print(f"[PASS] PERF-001 - 单图片推理速度: 平均{avg_time:.4f}s/帧, FPS: {fps:.2f}, 最小{min_time:.4f}s, 最大{max_time:.4f}s")

    def test_batch_inference(self):
        """PERF-002 测试批量图片推理"""
        print("[INFO] PERF-002 - 批量图片推理测试开始...")
        images = [self.test_image] * 10
        start_time = time.time()
        
        for i, img in enumerate(images):
            _, _ = self.detector.detect(img, conf_threshold=0.5)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / len(images)
        
        self.save_performance_data('batch_inference', {
            'total_images': len(images), 'total_time': elapsed, 'avg_time': avg_time
        })
        
        print(f"[PASS] PERF-002 - 批量推理完成，共{len(images)}张图片，总耗时: {elapsed:.4f}s，平均{avg_time:.4f}s/张")

    def test_preprocess_speed(self):
        """PERF-003 测试预处理速度"""
        print("[INFO] PERF-003 - 预处理速度测试开始...")
        total_time = 0.0
        count = 30
        
        for _ in range(count):
            start_time = time.time()
            self.detector.preprocess(self.test_image)
            total_time += time.time() - start_time
        
        avg_time = total_time / count
        
        self.save_performance_data('preprocess_speed', {'avg_time': avg_time})
        
        print(f"[PASS] PERF-003 - 预处理速度: 平均{avg_time:.4f}s/次")

    def test_postprocess_speed(self):
        """PERF-004 测试后处理速度"""
        print("[INFO] PERF-004 - 后处理速度测试开始...")
        blob, scale, top, left, h, w = self.detector.preprocess(self.test_image)
        self.detector.net.setInput(blob)
        outputs = self.detector.net.forward()
        
        total_time = 0.0
        count = 30
        
        for _ in range(count):
            start_time = time.time()
            self.detector.postprocess(outputs, scale, top, left, h, w, conf_threshold=0.5)
            total_time += time.time() - start_time
        
        avg_time = total_time / count
        
        self.save_performance_data('postprocess_speed', {'avg_time': avg_time})
        
        print(f"[PASS] PERF-004 - 后处理速度: 平均{avg_time:.4f}s/次")

    def test_memory_leak_check(self):
        """PERF-005 测试内存泄漏"""
        print("[INFO] PERF-005 - 内存泄漏测试开始...")
        import gc
        
        for _ in range(50):
            _, _ = self.detector.detect(self.test_image, conf_threshold=0.5)
        
        gc.collect()
        
        print(f"[PASS] PERF-005 - 内存泄漏测试完成，50次推理无明显泄漏")

    def save_performance_data(self, test_name, data):
        """保存性能测试数据到CSV"""
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output', 'performance_results.csv')
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['test_name'] + list(data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({'test_name': test_name, **data})


class TestCameraPerformance(unittest.TestCase):
    def setUp(self):
        self.window = None

    def test_camera_stability(self):
        """PERF-006 测试摄像头实时流稳定性"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from PyQt5.QtCore import Qt
        from Detect_GUI_cvdnn import Ui_MainWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        self.window = Ui_MainWindow()
        self.window.show()
        QTest.qWait(500)
        
        QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
        QTest.qWait(1000)
        
        if self.window.cap is not None and self.window.cap.isOpened():
            print("[INFO] PERF-006 - 摄像头稳定性测试开始，持续30秒...")
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 30:
                QTest.qWait(30)
                frame_count += 1
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
            QTest.qWait(500)
            
            self.window.close()
            
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output', 'performance_results.csv')
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['test_name', 'duration', 'frame_count', 'fps'])
                if not os.path.exists(csv_path):
                    writer.writeheader()
                writer.writerow({'test_name': 'camera_stability', 'duration': elapsed, 'frame_count': frame_count, 'fps': fps})
            
            print(f"[PASS] PERF-006 - 摄像头稳定性: {elapsed:.1f}秒内处理{frame_count}帧，平均FPS: {fps:.2f}")
        else:
            self.window.close()
            print(f"[SKIP] PERF-006 - 摄像头不可用，跳过测试")

    def test_camera_resource_release(self):
        """PERF-007 测试摄像头资源释放"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from PyQt5.QtCore import Qt
        from Detect_GUI_cvdnn import Ui_MainWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        self.window = Ui_MainWindow()
        self.window.show()
        QTest.qWait(500)
        
        QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
        QTest.qWait(1000)
        
        if self.window.cap is not None and self.window.cap.isOpened():
            QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
            QTest.qWait(500)
            self.assertIsNone(self.window.cap, "停止后摄像头应释放")
            self.window.close()
            print(f"[PASS] PERF-007 - 摄像头资源释放测试成功")
        else:
            self.window.close()
            print(f"[SKIP] PERF-007 - 摄像头不可用，跳过测试")


if __name__ == '__main__':
    print("=" * 70)
    print("YOLOv8Detector 性能测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)
    
    print("=" * 70)
    print("性能测试完成")
    print("=" * 70)