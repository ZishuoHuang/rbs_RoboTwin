import argparse
from envs import *
from config.generate_config import load_args
task = class_decorator("pick_diverse_bottles")
cfg = load_args("pick_diverse_bottles", "aloha-agilex_clean_50")
with open(cfg["save_path"]+"/seed.txt", "r") as f: seeds=[int(x) for x in f.read().split()]
task.setup_demo(now_ep_num=5, seed=seeds[5], **cfg)
traj = task.load_tran_data(5)
print(type(traj["left_joint_path"]))
print(traj["left_joint_path"][:])
