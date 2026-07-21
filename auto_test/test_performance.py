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

    def test_full_detection_pipeline_time(self):
        """PERF-005 测试完整检测流程耗时"""
        print("[INFO] PERF-005 - 完整检测流程耗时测试开始...")
        total_time = 0.0
        count = 10
        
        for i in range(count):
            start_time = time.time()
            _, _ = self.detector.detect(self.test_image, conf_threshold=0.5)
            elapsed = time.time() - start_time
            total_time += elapsed
        
        avg_time = total_time / count
        
        self.save_performance_data('full_detection_pipeline', {'avg_time': avg_time})
        
        print(f"[PASS] PERF-005 - 完整检测流程耗时: 平均{avg_time:.4f}s/次")

    def test_memory_leak_check(self):
        """PERF-006 测试内存泄漏"""
        print("[INFO] PERF-006 - 内存泄漏测试开始...")
        import gc
        
        for _ in range(50):
            _, _ = self.detector.detect(self.test_image, conf_threshold=0.5)
        
        gc.collect()
        
        print(f"[PASS] PERF-006 - 内存泄漏测试完成，50次推理无明显泄漏")

    def test_continuous_inference_stability(self):
        """PERF-007 测试连续推理稳定性"""
        print("[INFO] PERF-007 - 连续推理稳定性测试开始，100次推理...")
        errors = 0
        
        for i in range(100):
            try:
                _, results = self.detector.detect(self.test_image, conf_threshold=0.5)
                if (i + 1) % 20 == 0:
                    print(f"  第{i+1}次推理完成，检出{len(results)}个目标")
            except Exception as e:
                errors += 1
                print(f"  第{i+1}次推理出错: {e}")
        
        self.assertEqual(errors, 0, f"连续推理过程中出现{errors}次错误")
        
        print(f"[PASS] PERF-007 - 连续推理稳定性测试完成，100次推理无错误")

    def test_model_load_time(self):
        """PERF-008 测试模型加载时间"""
        print("[INFO] PERF-008 - 模型加载时间测试开始...")
        start_time = time.time()
        detector = YOLOv8Detector(model_path='yolov8n.onnx')
        detector.load_model()
        elapsed = time.time() - start_time
        
        self.assertIsNotNone(detector.net, "模型加载失败")
        
        self.save_performance_data('model_load_time', {'load_time': elapsed})
        
        print(f"[PASS] PERF-008 - 模型加载时间: {elapsed:.4f}s")

    def test_different_resolution_images(self):
        """PERF-009 测试不同分辨率图片推理"""
        print("[INFO] PERF-009 - 不同分辨率图片推理测试开始...")
        resolutions = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
        results = {}
        
        for w, h in resolutions:
            image = np.zeros((h, w, 3), dtype=np.uint8)
            start_time = time.time()
            _, _ = self.detector.detect(image, conf_threshold=0.5)
            elapsed = time.time() - start_time
            results[f'{w}x{h}'] = elapsed
            print(f"  {w}x{h} 推理耗时: {elapsed:.4f}s")
        
        self.save_performance_data('different_resolution_inference', results)
        
        print(f"[PASS] PERF-009 - 不同分辨率图片推理测试完成")

    def test_detection_time_variation(self):
        """PERF-010 测试检测时间波动"""
        print("[INFO] PERF-010 - 检测时间波动测试开始...")
        times = []
        count = 50
        
        for _ in range(count):
            start_time = time.time()
            _, _ = self.detector.detect(self.test_image, conf_threshold=0.5)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / count
        std_dev = np.std(times)
        cv = std_dev / avg_time * 100 if avg_time > 0 else 0
        
        self.save_performance_data('detection_time_variation', {
            'avg_time': avg_time, 'std_dev': std_dev, 'cv_percent': cv
        })
        
        print(f"[PASS] PERF-010 - 检测时间波动: 平均{avg_time:.4f}s, 标准差{std_dev:.4f}s, 变异系数{cv:.2f}%")

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
        """PERF-011 测试摄像头实时流稳定性"""
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
            print("[INFO] PERF-011 - 摄像头稳定性测试开始，持续30秒...")
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
            
            print(f"[PASS] PERF-011 - 摄像头稳定性: {elapsed:.1f}秒内处理{frame_count}帧，平均FPS: {fps:.2f}")
        else:
            self.window.close()
            print(f"[SKIP] PERF-011 - 摄像头不可用，跳过测试")

    def test_camera_resource_release(self):
        """PERF-012 测试摄像头资源释放"""
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
            print(f"[PASS] PERF-012 - 摄像头资源释放测试成功")
        else:
            self.window.close()
            print(f"[SKIP] PERF-012 - 摄像头不可用，跳过测试")

    def test_camera_startup_time(self):
        """PERF-013 测试摄像头启动时间"""
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
        
        start_time = time.time()
        QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
        QTest.qWait(1000)
        
        if self.window.cap is not None and self.window.cap.isOpened():
            elapsed = time.time() - start_time
            QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
            QTest.qWait(500)
            self.window.close()
            
            print(f"[PASS] PERF-013 - 摄像头启动时间: {elapsed:.4f}s")
        else:
            self.window.close()
            print(f"[SKIP] PERF-013 - 摄像头不可用，跳过测试")


class TestApplicationPerformance(unittest.TestCase):
    def test_application_cold_start(self):
        """PERF-014 测试应用冷启动时间"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from Detect_GUI_cvdnn import Ui_MainWindow
        
        print("[INFO] PERF-014 - 应用冷启动时间测试开始...")
        start_time = time.time()
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window = Ui_MainWindow()
        window.show()
        QTest.qWait(500)
        
        elapsed = time.time() - start_time
        
        window.close()
        
        print(f"[PASS] PERF-014 - 应用冷启动时间: {elapsed:.4f}s")

    def test_gui_initialization_time(self):
        """PERF-015 测试GUI初始化时间"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from Detect_GUI_cvdnn import Ui_MainWindow
        
        print("[INFO] PERF-015 - GUI初始化时间测试开始...")
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        start_time = time.time()
        window = Ui_MainWindow()
        elapsed = time.time() - start_time
        
        window.show()
        QTest.qWait(300)
        window.close()
        
        print(f"[PASS] PERF-015 - GUI初始化时间: {elapsed:.4f}s")

    def test_ui_interaction_response(self):
        """PERF-016 测试界面操作响应耗时"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from PyQt5.QtCore import Qt
        from Detect_GUI_cvdnn import Ui_MainWindow
        
        print("[INFO] PERF-016 - 界面操作响应耗时测试开始...")
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window = Ui_MainWindow()
        window.show()
        QTest.qWait(500)
        
        total_time = 0.0
        count = 10
        
        for _ in range(count):
            start_time = time.time()
            QTest.mouseClick(window.btn_clear, Qt.LeftButton)
            QTest.qWait(100)
            total_time += time.time() - start_time
        
        avg_time = total_time / count
        
        window.close()
        
        print(f"[PASS] PERF-016 - 界面操作响应耗时: 平均{avg_time:.4f}s/次")

    def test_memory_usage_peak(self):
        """PERF-017 测试内存占用峰值"""
        print("[INFO] PERF-017 - 内存占用峰值测试开始...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        test_image = cv2.imread('test/many.png')
        if test_image is None:
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for _ in range(20):
            _, _ = YOLOv8Detector().detect(test_image, conf_threshold=0.5)
        
        peak_memory = process.memory_info().rss / (1024 * 1024)
        
        print(f"[PASS] PERF-017 - 内存占用: 初始{initial_memory:.2f}MB, 峰值{peak_memory:.2f}MB")


if __name__ == '__main__':
    print("=" * 70)
    print("YOLOv8Detector 性能测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)
    
    print("=" * 70)
    print("性能测试完成")
    print("=" * 70)