import sys

# sys usage
sys.path.append('../leg_go1_cfg')
print(sys.path)

import os
path_1 = os.path.abspath('../leg_go1_cfg')
print("path_1 : ", path_1)

path_2 = os.path.abspath('..')
print("path_2 : ", path_2)

# 原来这里 print 的当前所在目录是命令行所在目录！！！
# 而不是该脚本所在目录！！ very confusing
path_3 = os.path.abspath('.')
print("path_3 : ", path_3)

from algo.ppo import PPO
print("successfully imported ppo")

