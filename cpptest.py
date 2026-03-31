import sys
import os

# 将编译产物的路径加入 Python 搜索路径
sys.path.append(os.path.join(os.getcwd(), "build/Release"))

try:
    import ba_core
    print("成功导入 ba_core 引擎！")
except ImportError as e:
    print(f"导入失败: {e}")