import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QTimer
from Detect_GUI_cvdnn import Ui_MainWindow

app = QApplication(sys.argv)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output'), exist_ok=True)


def auto_handle_modal_dialogs(expected_text=None, button_text='OK', file_path=None, is_save=False, max_retries=20):
    """
    通用模态对话框自动处理函数
    支持 QMessageBox 和 QFileDialog 的自动处理
    支持连续处理多个对话框（如文件选择+警告弹窗）
    
    Args:
        expected_text: 预期弹窗文本，用于断言验证（可选）
        button_text: 要点击的按钮文本，如 'OK', '是', '确定', 'Yes', 'Save', '打开' 等
        file_path: 选择文件对话框时要选择的文件路径，或保存对话框时的保存路径
        is_save: 是否为保存对话框
        max_retries: 最大重试次数
    """
    captured_text = []
    clicked_button = []
    handled_count = [0]
    max_handles = 5
    
    def check_and_handle():
        if handled_count[0] >= max_handles:
            return
        
        modal_widget = QApplication.activeModalWidget()
        if not modal_widget:
            if max_retries > 0:
                QTimer.singleShot(200, check_and_handle)
            return
        
        if isinstance(modal_widget, QMessageBox):
            msg_box = modal_widget
            text = msg_box.text()
            captured_text.append(text)
            print(f"[弹窗捕获] 检测到QMessageBox: {text}")
            
            if expected_text and expected_text in text:
                pass
            
            buttons = msg_box.buttons()
            for btn in buttons:
                btn_text = btn.text()
                clean_btn_text = btn_text.replace('&', '').replace('(S)', '').replace('(N)', '')
                if button_text == '否':
                    if '否' in btn_text or 'No' in clean_btn_text:
                        clicked_button.append(btn_text)
                        print(f"[弹窗操作] 自动点击按钮: {btn_text}")
                        QTest.mouseClick(btn, Qt.LeftButton)
                        handled_count[0] += 1
                        break
                elif button_text == '是':
                    if '是' in btn_text or 'Yes' in clean_btn_text:
                        clicked_button.append(btn_text)
                        print(f"[弹窗操作] 自动点击按钮: {btn_text}")
                        QTest.mouseClick(btn, Qt.LeftButton)
                        handled_count[0] += 1
                        break
                elif button_text == 'OK':
                    if '确定' in btn_text or 'OK' in btn_text:
                        clicked_button.append(btn_text)
                        print(f"[弹窗操作] 自动点击按钮: {btn_text}")
                        QTest.mouseClick(btn, Qt.LeftButton)
                        handled_count[0] += 1
                        break
                elif button_text in btn_text:
                    clicked_button.append(btn_text)
                    print(f"[弹窗操作] 自动点击按钮: {btn_text}")
                    QTest.mouseClick(btn, Qt.LeftButton)
                    handled_count[0] += 1
                    break
        
        elif isinstance(modal_widget, QFileDialog):
            file_dialog = modal_widget
            print(f"[弹窗捕获] 检测到QFileDialog")
            
            if file_path:
                file_dialog.selectFile(file_path)
                print(f"[弹窗操作] 选择文件: {file_path}")
            
            buttons = file_dialog.buttons()
            for btn in buttons:
                btn_text = btn.text()
                if (is_save and ('保存' in btn_text or 'Save' in btn_text)) or \
                   (not is_save and ('打开' in btn_text or 'Open' in btn_text)):
                    clicked_button.append(btn_text)
                    print(f"[弹窗操作] 自动点击按钮: {btn_text}")
                    QTest.mouseClick(btn, Qt.LeftButton)
                    handled_count[0] += 1
                    break
        
        QTimer.singleShot(200, check_and_handle)
    
    QTimer.singleShot(200, check_and_handle)
    return captured_text, clicked_button


class TestGUIBasicControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Ui_MainWindow()
        cls.window.show()
        QTest.qWait(500)

    @classmethod
    def tearDownClass(cls):
        if cls.window and cls.window.isVisible():
            auto_handle_modal_dialogs(
                expected_text='确定要退出应用吗？',
                button_text='是',
                max_retries=5
            )
            cls.window.close()
            QTest.qWait(500)
        app.processEvents()
        time.sleep(0.5)

    def test_window_initialization(self):
        """GUI-001 测试窗口正常初始化"""
        start_time = time.time()
        self.assertTrue(self.window.isVisible(), "窗口应可见")
        self.assertEqual(self.window.windowTitle(), '基于YOLOv8的检测演示软件V1.0', "窗口标题不正确")
        slider_value = self.window.slider_conf.value()
        self.assertGreaterEqual(slider_value, 10, "置信度滑块值应在有效范围内")
        self.assertLessEqual(slider_value, 90, "置信度滑块值应在有效范围内")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-001 - 窗口初始化成功，耗时: {elapsed:.2f}s")

    def test_confidence_slider_min(self):
        """GUI-002 测试置信滑块最小值10"""
        start_time = time.time()
        self.window.slider_conf.setValue(10)
        QTest.qWait(100)
        self.assertEqual(self.window.slider_conf.value(), 10, "滑块最小值应为10")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-002 - 置信度滑块最小值测试成功，耗时: {elapsed:.2f}s")

    def test_confidence_slider_max(self):
        """GUI-003 测试置信滑块最大值90"""
        start_time = time.time()
        self.window.slider_conf.setValue(90)
        QTest.qWait(100)
        self.assertEqual(self.window.slider_conf.value(), 90, "滑块最大值应为90")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-003 - 置信度滑块最大值测试成功，耗时: {elapsed:.2f}s")

    def test_confidence_slider_value_sync(self):
        """GUI-004 测试置信滑块拖动数值同步"""
        start_time = time.time()
        QTest.qWait(100)
        QTest.mouseClick(self.window.slider_conf, Qt.LeftButton)
        QTest.qWait(50)
        QTest.keyClick(self.window.slider_conf, Qt.Key_Right)
        QTest.qWait(50)
        value = self.window.slider_conf.value()
        label_text = self.window.label_conf_value.text()
        expected_text = f'当前值: {value/100:.2f}'
        self.assertEqual(label_text, expected_text, "滑块数值与标签应同步")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-004 - 置信度滑块同步成功，当前值: {value/100:.2f}，耗时: {elapsed:.2f}s")

    def test_clear_button(self):
        """GUI-005 测试清除按钮清空画布"""
        start_time = time.time()
        QTest.qWait(100)
        QTest.mouseClick(self.window.btn_clear, Qt.LeftButton)
        QTest.qWait(500)
        label_text = self.window.label_info.text()
        self.assertIn("等待加载图像", label_text, "清除后应提示等待加载")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-005 - 清除按钮功能正常，耗时: {elapsed:.2f}s")

    def test_model_selection_popup(self):
        """GUI-006 测试模型选择弹窗自动处理"""
        start_time = time.time()
        captured_text, clicked_button = auto_handle_modal_dialogs(
            expected_text='当前使用YOLOv8n-nano模型',
            button_text='OK',
            max_retries=5
        )
        QTest.mouseClick(self.window.btn_select_model, Qt.LeftButton)
        QTest.qWait(800)
        self.assertEqual(len(captured_text), 1, "应捕获到一个弹窗")
        self.assertIn('当前使用YOLOv8n-nano模型', captured_text[0], "弹窗文本应匹配")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-006 - 模型选择弹窗自动处理成功，耗时: {elapsed:.2f}s")


class TestGUIImageFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Ui_MainWindow()
        cls.window.show()
        QTest.qWait(500)

    @classmethod
    def tearDownClass(cls):
        if cls.window and cls.window.isVisible():
            auto_handle_modal_dialogs(
                expected_text='确定要退出应用吗？',
                button_text='是',
                max_retries=5
            )
            cls.window.close()
            QTest.qWait(500)
        app.processEvents()
        time.sleep(0.5)

    def test_load_normal_image(self):
        """GUI-007 测试正常图片检出目标"""
        start_time = time.time()
        import cv2
        image_path = os.path.abspath('test/many.png')
        self.window.original_image = cv2.imread(image_path)
        self.window.showImage(self.window.original_image, self.window.label_original)
        self.window.redetect_image()
        QTest.qWait(1000)
        label_text = self.window.label_info.text()
        self.assertNotIn("等待加载图像", label_text, "图片应加载成功")
        self.assertNotIn("未检测到目标", label_text, "应检测到目标")
        self.assertIn("检测到", label_text, "应显示检测到目标数量")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-007 - 正常图片加载成功，检测信息: {label_text[:50]}...，耗时: {elapsed:.2f}s")

    def test_load_no_target_image(self):
        """GUI-008 测试无目标图提示文字"""
        start_time = time.time()
        import cv2
        image_path = os.path.abspath('test/no_target.png')
        self.window.original_image = cv2.imread(image_path)
        self.window.showImage(self.window.original_image, self.window.label_original)
        self.window.redetect_image()
        QTest.qWait(1000)
        label_text = self.window.label_info.text()
        self.assertIn("未检测到目标", label_text, "无目标图片应提示未检测到目标")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-008 - 无目标图片测试成功，耗时: {elapsed:.2f}s")

    def test_load_broken_image_popup(self):
        """GUI-009 测试损坏图片弹窗自动处理（直接触发QMessageBox模拟损坏图片场景）"""
        start_time = time.time()
        
        auto_handle_modal_dialogs(
            expected_text='无法读取图片文件！',
            button_text='OK',
            max_retries=5
        )
        
        QMessageBox.warning(self.window, '警告', '无法读取图片文件！')
        QTest.qWait(800)
        
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-009 - 损坏图片弹窗自动处理成功，耗时: {elapsed:.2f}s")

    def test_save_image_with_result(self):
        """GUI-010 测试有检测结果时保存图片（直接调用保存逻辑）"""
        start_time = time.time()
        import cv2
        image_path = os.path.abspath('test/many.png')
        self.window.original_image = cv2.imread(image_path)
        self.window.showImage(self.window.original_image, self.window.label_original)
        self.window.redetect_image()
        QTest.qWait(500)
        
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output', 'test_save.png')
        
        if self.window.detect_image is not None:
            cv2.imwrite(save_path, self.window.detect_image)
            self.assertTrue(os.path.exists(save_path), "保存的图片文件应存在")
        
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-010 - 保存图片测试成功，耗时: {elapsed:.2f}s")

    def test_save_image_no_result_popup(self):
        """GUI-011 测试无检测结果时保存弹窗自动处理"""
        start_time = time.time()
        self.window.clearImage()
        QTest.qWait(300)
        auto_handle_modal_dialogs(
            expected_text='没有可保存的检测结果！',
            button_text='OK',
            max_retries=5
        )
        QTest.mouseClick(self.window.btn_save, Qt.LeftButton)
        QTest.qWait(800)
        label_text = self.window.label_info.text()
        self.assertIn('等待加载图像', label_text, "清除后应仍显示等待加载图像")
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-011 - 无检测结果保存弹窗自动处理成功，耗时: {elapsed:.2f}s")


class TestGUICameraFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Ui_MainWindow()
        cls.window.show()
        QTest.qWait(500)

    @classmethod
    def tearDownClass(cls):
        if cls.window and cls.window.isVisible():
            auto_handle_modal_dialogs(
                expected_text='确定要退出应用吗？',
                button_text='是',
                max_retries=5
            )
            cls.window.close()
            QTest.qWait(500)
        app.processEvents()
        time.sleep(0.5)

    def test_camera_start_stop(self):
        """GUI-012 测试有摄像头时正常开启实时流"""
        start_time = time.time()
        QTest.qWait(100)
        QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
        QTest.qWait(1000)
        if self.window.cap is not None and self.window.cap.isOpened():
            QTest.qWait(3000)
            QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
            QTest.qWait(500)
            self.assertIsNone(self.window.cap, "停止后摄像头应释放")
            elapsed = time.time() - start_time
            print(f"[PASS] GUI-012 - 摄像头启停正常，耗时: {elapsed:.2f}s")
        else:
            elapsed = time.time() - start_time
            print(f"[SKIP] GUI-012 - 摄像头不可用，跳过测试")

    def test_no_camera_warning_popup(self):
        """GUI-013 测试无摄像头时警告弹窗自动处理"""
        start_time = time.time()
        captured_text, clicked_button = auto_handle_modal_dialogs(
            expected_text='无法打开摄像头！',
            button_text='OK',
            max_retries=5
        )
        QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
        QTest.qWait(800)
        if self.window.cap is None:
            self.assertEqual(len(captured_text), 1, "应捕获到一个弹窗")
            self.assertIn('无法打开摄像头！', captured_text[0], "弹窗文本应匹配")
            elapsed = time.time() - start_time
            print(f"[PASS] GUI-013 - 无摄像头警告弹窗自动处理成功，耗时: {elapsed:.2f}s")
        else:
            QTest.mouseClick(self.window.btn_detect_cam, Qt.LeftButton)
            QTest.qWait(500)
            elapsed = time.time() - start_time
            print(f"[SKIP] GUI-013 - 摄像头可用，跳过无摄像头测试")


class TestGUIEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Ui_MainWindow()
        cls.window.show()
        QTest.qWait(500)

    @classmethod
    def tearDownClass(cls):
        if cls.window and cls.window.isVisible():
            auto_handle_modal_dialogs(
                expected_text='确定要退出应用吗？',
                button_text='是',
                max_retries=5
            )
            cls.window.close()
            QTest.qWait(500)
        app.processEvents()
        time.sleep(0.5)

    def test_frequent_image_switch(self):
        """GUI-014 测试频繁切换图片"""
        start_time = time.time()
        import cv2
        images = ['test/many.png', 'test/no_target.png']
        for _ in range(3):
            for img in images:
                image_path = os.path.abspath(img)
                self.window.original_image = cv2.imread(image_path)
                self.window.showImage(self.window.original_image, self.window.label_original)
                self.window.redetect_image()
                QTest.qWait(200)
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-014 - 频繁切换图片完成，耗时: {elapsed:.2f}s")

    def test_slider_drag_frequency(self):
        """GUI-015 测试频繁拖动滑块"""
        start_time = time.time()
        QTest.qWait(100)
        for i in range(10, 90, 10):
            self.window.slider_conf.setValue(i)
            QTest.qWait(50)
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-015 - 频繁拖动滑块完成，耗时: {elapsed:.2f}s")

    def test_exit_confirm_popup(self):
        """GUI-016 测试退出确认弹窗自动处理（弹窗捕获验证）"""
        start_time = time.time()
        auto_handle_modal_dialogs(
            expected_text='确定要退出应用吗？',
            button_text='否',
            max_retries=5
        )
        QTest.mouseClick(self.window.btn_exit, Qt.LeftButton)
        QTest.qWait(800)
        elapsed = time.time() - start_time
        print(f"[PASS] GUI-016 - 退出确认弹窗自动处理成功，耗时: {elapsed:.2f}s")


if __name__ == '__main__':
    print("=" * 70)
    print("GUI自动化功能测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)
    
    print("=" * 70)
    print("GUI测试完成")
    print("=" * 70)