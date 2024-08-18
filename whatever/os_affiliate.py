import os # 做些与操作系统相关的操作

# nt = windows; posix = linux
print("device : ", os.name)

# 返回当前工作目录
print("current working directory :", os.getcwd())

# 返回文件的绝对路径
print("absolute path :", os.path.abspath('os_affiliate.py'))

# 返回文件名
print("file name :", os.path.basename('./os_affiliates.py')) # 事实证明这个路径瞎编都可以，它应该就是取了最后一个下划线后面的那部分

# 把目录和文件名合成一个路径
print("path :", os.path.join('/', 'test', 'test.txt'))

# 把路径拆成目录部分和文件名部分
print("path :", os.path.split('/test/test.txt'))