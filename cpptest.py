import sys
import os

try:
    from build.Release import ba_core
    print("成功导入 ba_core 引擎！")
except ImportError as e:
    print(f"导入失败: {e}")