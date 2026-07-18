import sys
import os
import time
import unittest
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from Detect_GUI_cvdnn import YOLOv8Detector

os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output'), exist_ok=True)


def capture_screenshot(test_name, window=None):
    """测试失败时捕获截图"""
    try:
        if window:
            screenshot = window.grab()
            screenshot.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output', f'{test_name}_failed.png'))
            print(f"[截图保存] {test_name} 失败截图已保存")
    except Exception as e:
        print(f"[截图失败] {test_name} 截图保存失败: {e}")


class TestYOLOv8DetectorModelLoading(unittest.TestCase):
    def test_load_valid_model(self):
        """UNIT-001 测试加载正常ONNX模型"""
        start_time = time.time()
        detector = YOLOv8Detector(model_path='yolov8n.onnx')
        detector.load_model()
        elapsed = time.time() - start_time
        self.assertIsNotNone(detector.net, "有效模型加载失败")
        print(f"[PASS] UNIT-001 - 有效模型加载成功，耗时: {elapsed:.2f}s")

    def test_load_valid_model_auto(self):
        """UNIT-002 测试自动加载模型"""
        start_time = time.time()
        detector = YOLOv8Detector()
        img = cv2.imread('test/many.png')
        _, results = detector.detect(img)
        elapsed = time.time() - start_time
        self.assertIsNotNone(detector.net, "自动加载模型失败")
        print(f"[PASS] UNIT-002 - 自动加载模型成功，耗时: {elapsed:.2f}s")

    def test_load_invalid_model(self):
        """UNIT-003 测试加载损坏模型文件"""
        start_time = time.time()
        detector = YOLOv8Detector(model_path='invalid.onnx')
        detector.load_model()
        elapsed = time.time() - start_time
        self.assertIsNone(detector.net, "损坏模型应加载失败")
        print(f"[PASS] UNIT-003 - 损坏模型正确处理，耗时: {elapsed:.2f}s")

    def test_load_nonexistent_model(self):
        """UNIT-004 测试加载不存在的模型文件"""
        start_time = time.time()
        detector = YOLOv8Detector(model_path='/nonexistent/path/model.onnx')
        detector.load_model()
        elapsed = time.time() - start_time
        self.assertIsNone(detector.net, "不存在的模型应加载失败")
        print(f"[PASS] UNIT-004 - 不存在模型正确处理，耗时: {elapsed:.2f}s")

    def test_model_input_shape(self):
        """UNIT-005 测试模型输入形状"""
        start_time = time.time()
        detector = YOLOv8Detector()
        self.assertEqual(detector.input_shape, (640, 640), "输入形状应为640x640")
        elapsed = time.time() - start_time
        print(f"[PASS] UNIT-005 - 模型输入形状正确，耗时: {elapsed:.2f}s")

    def test_class_names_count(self):
        """UNIT-006 测试类别名称数量"""
        start_time = time.time()
        detector = YOLOv8Detector()
        self.assertEqual(len(detector.class_names), 80, "类别数量应为80")
        elapsed = time.time() - start_time
        print(f"[PASS] UNIT-006 - 类别名称数量正确，耗时: {elapsed:.2f}s")


class TestYOLOv8DetectorPreprocess(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv8Detector()

    def test_preprocess_normal_image(self):
        """UNIT-007 测试标准图像预处理"""
        start_time = time.time()
        image = cv2.imread('test/many.png')
        self.assertIsNotNone(image, "测试图片读取失败")
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "预处理输出形状错误")
        self.assertGreater(scale, 0, "缩放比例应为正数")
        print(f"[PASS] UNIT-007 - 标准图像处理成功，耗时: {elapsed:.2f}s")

    def test_preprocess_large_image(self):
        """UNIT-008 测试超大图像预处理"""
        start_time = time.time()
        image = cv2.imread('test/big_4k.png')
        if image is None:
            self.skipTest("4K测试图片不存在")
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "超大图像预处理输出形状错误")
        self.assertLess(scale, 1.0, "超大图像缩放比例应小于1")
        print(f"[PASS] UNIT-008 - 超大图像处理成功，耗时: {elapsed:.2f}s")

    def test_preprocess_small_image(self):
        """UNIT-009 测试极小图像预处理"""
        start_time = time.time()
        image = cv2.imread('test/small_100x100.png')
        if image is None:
            self.skipTest("小尺寸测试图片不存在")
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "极小图像预处理输出形状错误")
        self.assertGreater(scale, 1.0, "极小图像缩放比例应大于1")
        print(f"[PASS] UNIT-009 - 极小图像处理成功，耗时: {elapsed:.2f}s")

    def test_preprocess_gray_image(self):
        """UNIT-010 测试灰度图像预处理"""
        start_time = time.time()
        image = cv2.imread('test/gray_img.png')
        if image is None:
            self.skipTest("灰度测试图片不存在")
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "灰度图像预处理输出形状错误")
        print(f"[PASS] UNIT-010 - 灰度图像处理成功，耗时: {elapsed:.2f}s")

    def test_preprocess_square_image(self):
        """UNIT-011 测试正方形图像预处理"""
        start_time = time.time()
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "正方形图像预处理输出形状错误")
        print(f"[PASS] UNIT-011 - 正方形图像处理成功，耗时: {elapsed:.2f}s")

    def test_preprocess_stretch_image(self):
        """UNIT-012 测试拉伸图像预处理"""
        start_time = time.time()
        image = np.zeros((240, 960, 3), dtype=np.uint8)
        blob, scale, top, left, h, w = self.detector.preprocess(image)
        elapsed = time.time() - start_time
        self.assertEqual(blob.shape, (1, 3, 640, 640), "拉伸图像预处理输出形状错误")
        print(f"[PASS] UNIT-012 - 拉伸图像处理成功，耗时: {elapsed:.2f}s")


class TestYOLOv8DetectorDetect(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv8Detector()

    def test_detect_with_target(self):
        """UNIT-013 测试含目标图片正常检出"""
        start_time = time.time()
        image = cv2.imread('test/many.png')
        self.assertIsNotNone(image, "测试图片读取失败")
        _, results = self.detector.detect(image, conf_threshold=0.5)
        elapsed = time.time() - start_time
        self.assertIsInstance(results, list, "检测结果应为列表")
        self.assertGreater(len(results), 0, "应检测到目标")
        detected_classes = [r['class_name'] for r in results]
        self.assertIn('cat', detected_classes, "应检测到cat类别")
        self.assertIn('dog', detected_classes, "应检测到dog类别")
        print(f"[PASS] UNIT-013 - 含目标图片检测成功，检出{len(results)}个目标({detected_classes})，耗时: {elapsed:.2f}s")

    def test_detect_no_target(self):
        """UNIT-014 测试无目标图片返回空列表"""
        start_time = time.time()
        image = cv2.imread('test/no_target.png')
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, results = self.detector.detect(image, conf_threshold=0.5)
        elapsed = time.time() - start_time
        self.assertEqual(len(results), 0, "无目标图片应返回空列表")
        print(f"[PASS] UNIT-014 - 无目标图片检测成功，耗时: {elapsed:.2f}s")

    def test_detect_confidence_threshold(self):
        """UNIT-015 测试不同置信阈值过滤效果"""
        start_time = time.time()
        image = cv2.imread('test/many.png')
        self.assertIsNotNone(image, "测试图片读取失败")
        _, results_low = self.detector.detect(image, conf_threshold=0.1)
        _, results_high = self.detector.detect(image, conf_threshold=0.9)
        elapsed = time.time() - start_time
        self.assertGreaterEqual(len(results_low), len(results_high), "低阈值检出数量应大于等于高阈值")
        print(f"[PASS] UNIT-015 - 阈值过滤验证成功，低阈值检出{len(results_low)}个，高阈值检出{len(results_high)}个，耗时: {elapsed:.2f}s")

    def test_detect_return_original_image(self):
        """UNIT-016 测试返回原始图像"""
        start_time = time.time()
        image = cv2.imread('test/many.png')
        self.assertIsNotNone(image, "测试图片读取失败")
        returned_image, _ = self.detector.detect(image)
        elapsed = time.time() - start_time
        self.assertEqual(returned_image.shape, image.shape, "返回图像形状应与输入一致")
        print(f"[PASS] UNIT-016 - 返回原始图像测试成功，耗时: {elapsed:.2f}s")

    def test_detect_result_structure(self):
        """UNIT-017 测试检测结果结构"""
        start_time = time.time()
        image = cv2.imread('test/many.png')
        self.assertIsNotNone(image, "测试图片读取失败")
        _, results = self.detector.detect(image)
        elapsed = time.time() - start_time
        if len(results) > 0:
            result = results[0]
            self.assertIn('box', result, "结果应包含box字段")
            self.assertIn('score', result, "结果应包含score字段")
            self.assertIn('class_id', result, "结果应包含class_id字段")
            self.assertIn('class_name', result, "结果应包含class_name字段")
            self.assertEqual(len(result['box']), 4, "box应包含4个坐标")
        print(f"[PASS] UNIT-017 - 检测结果结构验证成功，耗时: {elapsed:.2f}s")


class TestYOLOv8DetectorPostprocess(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv8Detector()

    def test_postprocess_nms_filtering(self):
        """UNIT-018 测试NMS抑制重复框"""
        start_time = time.time()
        outputs = np.zeros((1, 84, 100), dtype=np.float32)
        outputs[0, :4, 0] = [320, 320, 100, 100]
        outputs[0, 4, 0] = 0.9
        outputs[0, 5, 0] = 0.8
        outputs[0, :4, 1] = [325, 325, 100, 100]
        outputs[0, 4, 1] = 0.8
        outputs[0, 5, 1] = 0.7
        outputs[0, :4, 2] = [100, 100, 50, 50]
        outputs[0, 4, 2] = 0.95
        outputs[0, 5, 2] = 0.9
        results = self.detector.postprocess(outputs, 1.0, 0, 0, 640, 640)
        elapsed = time.time() - start_time
        self.assertLessEqual(len(results), 3, "NMS后结果数量应小于等于原始数量")
        self.assertGreaterEqual(len(results), 2, "NMS应保留非重复框")
        print(f"[PASS] UNIT-018 - NMS过滤成功，剩余{len(results)}个检测框，耗时: {elapsed:.2f}s")

    def test_postprocess_boundary_clipping(self):
        """UNIT-019 测试越界坐标自动裁剪"""
        start_time = time.time()
        outputs = np.zeros((1, 84, 100), dtype=np.float32)
        outputs[0, :4, 0] = [320, 320, 800, 800]
        outputs[0, 4, 0] = 0.9
        outputs[0, 5, 0] = 0.8
        results = self.detector.postprocess(outputs, 1.0, 0, 0, 480, 640)
        elapsed = time.time() - start_time
        self.assertEqual(len(results), 1, "应检测到一个目标")
        x1, y1, x2, y2 = results[0]['box']
        self.assertGreaterEqual(x1, 0, "x1应大于等于0")
        self.assertGreaterEqual(y1, 0, "y1应大于等于0")
        self.assertLessEqual(x2, 640, "x2应小于等于图像宽度")
        self.assertLessEqual(y2, 480, "y2应小于等于图像高度")
        print(f"[PASS] UNIT-019 - 越界坐标裁剪成功，裁剪后框: [{x1},{y1},{x2},{y2}]，耗时: {elapsed:.2f}s")

    def test_postprocess_invalid_class_filter(self):
        """UNIT-020 测试无效类别过滤"""
        start_time = time.time()
        outputs = np.zeros((1, 84, 100), dtype=np.float32)
        outputs[0, :4, 0] = [320, 320, 100, 100]
        outputs[0, 4, 0] = 0.9
        outputs[0, 5:, 0] = np.zeros(79)
        outputs[0, 5, 0] = 1.0
        results = self.detector.postprocess(outputs, 1.0, 0, 0, 640, 640)
        elapsed = time.time() - start_time
        self.assertEqual(len(results), 1, "有效类别应被保留")
        self.assertEqual(results[0]['class_id'], 0, "类别ID应为0(person)")
        print(f"[PASS] UNIT-020 - 有效类别过滤成功，类别ID: {results[0]['class_id']}，耗时: {elapsed:.2f}s")

    def test_postprocess_empty_output(self):
        """UNIT-021 测试空输出返回空列表（所有置信度低于阈值）"""
        start_time = time.time()
        outputs = np.zeros((1, 84, 100), dtype=np.float32)
        outputs[0, :4, 0] = [320, 320, 100, 100]
        outputs[0, 4, 0] = 0.1
        outputs[0, 5:, 0] = np.zeros(79)
        outputs[0, 5, 0] = 0.1
        results = self.detector.postprocess(outputs, 1.0, 0, 0, 640, 640, conf_threshold=0.5)
        elapsed = time.time() - start_time
        self.assertEqual(len(results), 0, "置信度低于阈值应返回空列表")
        print(f"[PASS] UNIT-021 - 低置信度输出处理成功（返回空列表），耗时: {elapsed:.2f}s")


class TestYOLOv8DetectorDrawResults(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv8Detector()

    def test_draw_empty_results(self):
        """UNIT-022 测试空结果无报错"""
        start_time = time.time()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        drawn_image = self.detector.draw_results(image, [])
        elapsed = time.time() - start_time
        self.assertEqual(drawn_image.shape, image.shape, "绘制后图像尺寸应保持一致")
        np.testing.assert_array_equal(drawn_image, image, "空结果绘制应与原图一致")
        print(f"[PASS] UNIT-022 - 空结果绘制成功，耗时: {elapsed:.2f}s")

    def test_draw_multiple_classes(self):
        """UNIT-023 测试多类别绘制对应颜色标注"""
        start_time = time.time()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [
            {'box': [100, 100, 200, 200], 'score': 0.9, 'class_id': 0, 'class_name': 'person'},
            {'box': [300, 300, 400, 400], 'score': 0.8, 'class_id': 2, 'class_name': 'car'}
        ]
        drawn_image = self.detector.draw_results(image, results)
        elapsed = time.time() - start_time
        self.assertEqual(drawn_image.shape, image.shape, "绘制后图像尺寸应保持一致")
        self.assertFalse(np.array_equal(drawn_image, image), "绘制后图像应与原图不同")
        print(f"[PASS] UNIT-023 - 多类别绘制成功，耗时: {elapsed:.2f}s")

    def test_draw_large_box(self):
        """UNIT-024 测试大尺寸检测框绘制"""
        start_time = time.time()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{'box': [0, 0, 640, 480], 'score': 0.9, 'class_id': 0, 'class_name': 'person'}]
        drawn_image = self.detector.draw_results(image, results)
        elapsed = time.time() - start_time
        self.assertEqual(drawn_image.shape, image.shape, "绘制后图像尺寸应保持一致")
        print(f"[PASS] UNIT-024 - 大尺寸检测框绘制成功，耗时: {elapsed:.2f}s")

    def test_draw_small_box(self):
        """UNIT-025 测试小尺寸检测框绘制"""
        start_time = time.time()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{'box': [10, 10, 20, 20], 'score': 0.9, 'class_id': 0, 'class_name': 'person'}]
        drawn_image = self.detector.draw_results(image, results)
        elapsed = time.time() - start_time
        self.assertEqual(drawn_image.shape, image.shape, "绘制后图像尺寸应保持一致")
        print(f"[PASS] UNIT-025 - 小尺寸检测框绘制成功，耗时: {elapsed:.2f}s")


if __name__ == '__main__':
    print("=" * 70)
    print("YOLOv8Detector 单元测试")
    print("=" * 70)
    unittest.main(verbosity=2)
    print("=" * 70)
    print("单元测试完成")
    print("=" * 70)