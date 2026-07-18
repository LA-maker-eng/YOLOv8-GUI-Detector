# YOLOv8 PyQt5 视觉检测上位机 - 自动化测试运维手册

## 一、项目目录结构

```
YOLOv8_PYQT5_GUI-main/
├── auto_test/                    # 测试脚本目录
│   ├── __init__.py
│   ├── test_unit_core.py         # 单元测试脚本
│   ├── test_gui_flow.py          # GUI自动化测试脚本
│   └── test_performance.py       # 性能测试脚本
├── test/                         # 测试素材目录
│   ├── many.png                  # 含多目标测试图片
│   ├── no_target.png             # 无目标测试图片
│   ├── broken.png                # 损坏图片
│   ├── big_4k.png                # 大尺寸图片（可选）
│   ├── small_100x100.png         # 小尺寸图片（可选）
│   └── gray_img.png              # 灰度图片（可选）
├── test_output/                  # 测试输出目录
│   ├── performance_results.csv   # 性能测试结果
│   ├── test_save.png             # 测试保存图片
│   └── [测试名]_failed.png       # 失败截图
├── Detect_GUI_cvdnn.py          # 主程序
├── yolov8n.onnx                 # 模型文件
├── test_report.md               # 测试报告
├── test_cases.md                # 测试用例清单
├── test_manual.md               # 运维手册
├── defect_log.md                # 缺陷台账
└── .gitignore                  # Git忽略配置
```

## 二、一键执行命令

### 2.1 运行全部测试

```powershell
cd C:\Users\liang\Desktop\YOLOv8_PYQT5_GUI-main
.\venv\Scripts\Activate.ps1
python -m unittest auto_test/*.py -v
```

### 2.2 单独运行单元测试

```powershell
python -m unittest auto_test.test_unit_core -v
```

### 2.3 单独运行GUI测试

```powershell
python -m unittest auto_test.test_gui_flow -v
```

### 2.4 单独运行性能测试

```powershell
python -m unittest auto_test.test_performance -v
```

### 2.5 运行特定测试类

```powershell
python -m unittest auto_test.test_unit_core.TestYOLOv8DetectorModelLoading -v
```

### 2.6 运行特定测试用例

```powershell
python -m unittest auto_test.test_unit_core.TestYOLOv8DetectorModelLoading.test_load_valid_model -v
```

## 三、环境搭建步骤

### 3.1 Python环境

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install pyqt5 opencv-python numpy
```

### 3.2 测试素材准备

将测试图片放入 `test/` 目录：
- `many.png`: 包含多个检测目标的图片
- `no_target.png`: 纯白或无目标图片
- `broken.png`: 损坏的图片文件

### 3.3 模型文件

确保 `yolov8n.onnx` 文件存在于项目根目录。

## 四、常见报错解决方案

### 4.1 模型加载失败

**错误信息**: `Can't read ONNX file`

**解决方案**:
1. 检查 `yolov8n.onnx` 文件是否存在
2. 确认模型文件未损坏
3. 检查文件路径是否正确

### 4.2 测试图片读取失败

**错误信息**: `测试图片读取失败`

**解决方案**:
1. 检查测试图片是否存在于 `test/` 目录
2. 确认图片格式正确 (PNG/JPG)
3. 检查文件路径是否正确

### 4.3 摄像头不可用

**错误信息**: `无法打开摄像头！`

**解决方案**:
1. 检查摄像头硬件连接
2. 确认摄像头未被其他程序占用
3. 检查摄像头权限

### 4.4 Qt弹窗阻塞测试

**错误信息**: 测试卡住不继续

**解决方案**:
1. 确保使用 `auto_close_message_box()` 函数处理弹窗
2. 调整 `delay_ms` 参数确保弹窗已出现
3. 检查弹窗按钮文本是否匹配

### 4.5 测试超时

**解决方案**:
1. 增加 `QTest.qWait()` 的等待时间
2. 检查是否有未处理的弹窗
3. 确认硬件资源充足

## 五、新增用例编写规范

### 5.1 单元测试编写规范

```python
def test_xxx(self):
    """UNIT-XXX 测试名称"""
    start_time = time.time()
    # 测试逻辑
    # 断言
    elapsed = time.time() - start_time
    print(f"[PASS] UNIT-XXX - 测试名称，耗时: {elapsed:.2f}s")
```

### 5.2 GUI测试编写规范

```python
def test_xxx(self):
    """GUI-XXX 测试名称"""
    start_time = time.time()
    # 如果涉及弹窗，先设置自动处理
    captured_text, clicked_button = auto_close_message_box(
        expected_text='弹窗文本',
        button_text='OK',
        delay_ms=200
    )
    # 执行操作
    QTest.mouseClick(self.window.btn_xxx, Qt.LeftButton)
    QTest.qWait(500)
    # 断言
    elapsed = time.time() - start_time
    print(f"[PASS] GUI-XXX - 测试名称，耗时: {elapsed:.2f}s")
```

### 5.3 性能测试编写规范

```python
def test_xxx(self):
    """PERF-XXX 测试名称"""
    print("[INFO] PERF-XXX - 测试名称开始...")
    # 性能测试逻辑
    # 保存数据
    self.save_performance_data('test_name', {'key': value})
    print(f"[PASS] PERF-XXX - 测试结果")
```

### 5.4 用例编号规则

| 前缀 | 模块 | 范围 |
|------|------|------|
| UNIT | 单元测试 | 001-999 |
| GUI | GUI测试 | 001-999 |
| PERF | 性能测试 | 001-999 |

## 六、测试输出说明

### 6.1 控制台输出

测试运行时会输出以下信息：
- `[PASS]`: 测试通过
- `[FAIL]`: 测试失败
- `[SKIP]`: 测试跳过
- `[INFO]`: 测试信息
- `[弹窗捕获]`: 自动捕获弹窗
- `[弹窗操作]`: 自动点击按钮

### 6.2 文件输出

| 文件 | 说明 |
|------|------|
| test_output/performance_results.csv | 性能测试数据 |
| test_output/test_save.png | 测试保存图片 |
| test_output/[测试名]_failed.png | 失败截图 |
| test_report.md | 测试报告 |

## 七、注意事项

1. **运行时间**: 全部测试约需90-120秒
2. **GUI测试**: 运行时会弹出测试窗口，测试完成后自动关闭
3. **摄像头测试**: 需要有可用摄像头，否则自动跳过
4. **弹窗处理**: 所有弹窗自动处理，无需人工干预
5. **权限要求**: 需要摄像头访问权限和文件读写权限