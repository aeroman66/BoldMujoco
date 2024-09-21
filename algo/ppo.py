import os
import sys

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from module import ActorCritic