# 这里为了方便演示 __init__ 的方便导入功能，将原先的 helper 脚本进行拆分
from .helpers import class_to_dict
from .utils_go1 import split_and_pad_trajectories
from .normalizer import EmpiricalNormalization