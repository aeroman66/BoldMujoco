# 参考 legged-gym 的base_config 进行编写
# 这个脚本的主要功能是定义一个基础配置类 BaseConfig。
# 它实现了一个初始化方法，可以递归地初始化所有成员类。
# 这个类作为其他具体机器人和环境配置类的基类，提供了一个通用的配置结构和初始化机制。

import inspect

class BaseConfig:
    def __init__(self) -> None:
        """
        Initializes all member classes recursively. Ignores all names starting with '__' (built-in methods).
        """
        self.init_member_classes(self)

    @staticmethod # 这个装饰器的功能还不是很懂
    def init_member_classes(obj):
        # iterate over all classes' names
        for key in dir(obj):
            # disregard built-in attributes
            if key.startswith('__'):
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                i_var = var() # 这步应该相当于让类调用自己的默认构造函数了
                setattr(obj, key, i_var) # 然后将这个类设置为属性值
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var) # 防止类内有类，像是 cmd_range