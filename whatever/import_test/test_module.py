def laugh():
    print("laugh") # 但是问题是这个脚本并没有被执行

print(__name__)

# 这个的意思就是，只有在当前文件中运行时，才会执行下面的语句，在其他文件中导入时，不会执行下面的语句
if __name__ == '__main__':
    laugh()