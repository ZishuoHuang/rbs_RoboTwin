from envs import *
from config.generate_config import load_args
task = class_decorator("pick_diverse_bottles")
cfg = load_args("pick_diverse_bottles", "aloha-agilex_clean_50")
with open(cfg["save_path"]+"/seed.txt", "r") as f: seeds=[int(x) for x in f.read().split()]

def measure_len():
    task.setup_demo(now_ep_num=5, seed=seeds[5], **cfg)
    traj = task.load_tran_data(5)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)
    c = 0
    orig = task._update_render
    def dummy():
        nonlocal c; c += 1
    task._update_render = dummy
    try: task.play_once(); task.viewer.close()
    except Exception as e: print(e)
    return c

print("Pass 1:", measure_len())
try: task.close_env(clear_cache=True)
except: pass

task = class_decorator("pick_diverse_bottles")
print("Pass 2:", measure_len())
