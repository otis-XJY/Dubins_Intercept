"""
python_Dubins 项目主包
Dubins-A*算法的Python实现，用于路径规划
"""

__version__ = "1.0.0"
__author__ = "CASIA"

# 导出主要模块
from . import Draw
from . import DDebug
from . import A_dubins
from . import intercept
from . import main

__all__ = ['Draw', 'DDebug', 'A_dubins', 'intercept', 'main']