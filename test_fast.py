import argparse
from envs import *
from config.generate_config import load_args
task = class_decorator("pick_diverse_bottles")
cfg = load_args("pick_diverse_bottles", "aloha-agilex_clean_50")
with open(cfg["save_path"]+"/seed.txt", "r") as f: seeds=[int(x) for x in f.read().split()]
task.setup_demo(now_ep_num=5, seed=seeds[5], **cfg)
traj = task.load_tran_data(5)
cfg["left_joint_path"] = traj["left_joint_path"]
cfg["right_joint_path"] = traj["right_joint_path"]
task.set_path_lst(cfg)
count = 0
orig = task._update_render
def my_up():
    global count
    count += 1
    orig()
task._update_render = my_up
try:
    task.play_once()
except Exception: pass
print("TOTAL FRAMES:", count)
