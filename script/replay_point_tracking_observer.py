import argparse
import importlib
import os
import sys

import numpy as np
import sapien.core as sapien
import yaml

sys.path.append("./")
from envs import *


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_args(task_name, task_config):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    emb = args.get("embodiment")

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_emb_file(name):
        robot_file = emb_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(emb) == 1:
        args["left_robot_file"] = get_emb_file(emb[0])
        args["right_robot_file"] = get_emb_file(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = get_emb_file(emb[0])
        args["right_robot_file"] = get_emb_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    return args


def get_entity_scene_id(entity):
    fn = getattr(entity, "get_per_scene_id", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    val = getattr(entity, "per_scene_id", None)
    if val is not None:
        return int(val)

    # PhysxArticulationLinkComponent often stores the render entity in `.entity`.
    # Segmentation actor ids usually match this entity's per_scene_id.
    ent = getattr(entity, "entity", None)
    ent_val = getattr(ent, "per_scene_id", None) if ent is not None else None
    if ent_val is not None:
        return int(ent_val)

    return None


def build_scene_entity_map(task_env):
    entity_map = {}

    for actor in task_env.scene.get_all_actors():
        sid = get_entity_scene_id(actor)
        if sid is not None:
            entity_map[sid] = actor

    for link in task_env.robot.left_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    for link in task_env.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    return entity_map


def get_background_scene_ids(task_env):
    background_names = {"ground", "wall", "table"}
    background_ids = set()
    for actor in task_env.scene.get_all_actors():
        if actor.get_name() in background_names:
            sid = get_entity_scene_id(actor)
            if sid is not None:
                background_ids.add(int(sid))
    return background_ids


def create_markers(scene, n, radius):
    markers = []
    colors = []
    for i in range(n):
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius)
        marker = builder.build_kinematic(name=f"obs_track_marker_{i}")
        markers.append(marker)
        colors.append([
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
        ])
    return markers, colors


def colorize_markers(markers, colors):
    for marker, c in zip(markers, colors):
        try:
            render_body = marker.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                shape.material.set_base_color([c[0], c[1], c[2], 1.0])
        except Exception:
            continue


def resolve_source_camera(task, camera_name):
    if camera_name == "world_camera":
        camera_name = "world_camera1"

    if hasattr(task.cameras, camera_name):
        return getattr(task.cameras, camera_name)

    obs = getattr(task.cameras, "observer_camera", None)
    if camera_name == "observer_camera" and obs is not None:
        return obs

    world1 = getattr(task.cameras, "world_camera1", None)
    world2 = getattr(task.cameras, "world_camera2", None)
    if camera_name == "world_camera1" and world1 is not None:
        return world1
    if camera_name == "world_camera2" and world2 is not None:
        return world2

    raise KeyError(
        f"Unknown camera '{camera_name}'. Available in this script: "
        "observer_camera, world_camera1, world_camera2, head_camera, left_camera, right_camera"
    )


def point_texture_to_world(camera, position_texture, u, v):
    # Position texture stores the camera-space position in OpenGL convention.
    p_cam = position_texture[v, u, :3].astype(np.float32)
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32)
    T_c2w = camera.get_model_matrix().astype(np.float32)
    return (T_c2w @ p_h)[:3].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Replay point tracking from observer_camera viewpoint")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument(
        "--camera",
        type=str,
        default="observer_camera",
        help="Source camera for frame-0 sampling: observer_camera, world_camera1, world_camera2, head_camera, left_camera, right_camera",
    )
    parser.add_argument("--num-points", type=int, default=5000)
    parser.add_argument("--robot-ratio", type=float, default=0.45)
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--include-background", action="store_true", help="Keep table/wall/ground points as well")
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)

    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False
    cfg["render_freq"] = 1

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]

    ep = args.episode
    if ep < 0 or ep >= len(seeds):
        raise RuntimeError(f"episode out of range: {ep}, available 0..{len(seeds)-1}")

    task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg)

    traj = task.load_tran_data(ep)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    # Build the scene once and capture the observer camera view.
    task.get_obs()
    source_camera = resolve_source_camera(task, args.camera)
    source_camera.take_picture()

    position = source_camera.get_picture("Position")
    segmentation = source_camera.get_picture("Segmentation")

    seg_raw = segmentation[..., 1].astype(np.int32)
    valid = position[..., 3] < 1
    if not args.include_background:
        valid &= seg_raw > 0

    visible_ids = np.unique(seg_raw[valid])
    entity_map = build_scene_entity_map(task)
    background_ids = get_background_scene_ids(task)
    robot_ids = set()
    for link in task.robot.left_entity.get_links() + task.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            robot_ids.add(int(sid))

    if args.include_background:
        filtered_visible_ids = [int(s) for s in visible_ids]
    else:
        filtered_visible_ids = [int(s) for s in visible_ids if int(s) not in background_ids]

    robot_visible = [int(s) for s in filtered_visible_ids if int(s) in robot_ids]
    object_visible = [int(s) for s in filtered_visible_ids if int(s) not in robot_ids]

    def sample_for_ids(ids, quota):
        if quota <= 0 or len(ids) == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        per_id = max(1, quota // len(ids))
        sampled_u = []
        sampled_v = []
        for seg_id in ids:
            mask = valid & (seg_raw == seg_id)
            vv, uu = np.where(mask)
            if len(uu) == 0:
                continue
            take = min(len(uu), per_id)
            pick = np.random.choice(len(uu), take, replace=False)
            sampled_u.append(uu[pick])
            sampled_v.append(vv[pick])
        if len(sampled_u) == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.concatenate(sampled_u), np.concatenate(sampled_v)

    target_robot = int(round(args.num_points * args.robot_ratio))
    target_object = max(0, args.num_points - target_robot)

    uu_robot, vv_robot = sample_for_ids(robot_visible, target_robot)
    uu_obj, vv_obj = sample_for_ids(object_visible, target_object)
    uu = np.concatenate([uu_robot, uu_obj])
    vv = np.concatenate([vv_robot, vv_obj])

    if len(uu) == 0:
        vv, uu = np.where(valid)
        if len(uu) == 0:
            raise RuntimeError("No valid observer-camera pixels found")
        if len(uu) > args.num_points:
            sel = np.random.choice(len(uu), args.num_points, replace=False)
            uu = uu[sel]
            vv = vv[sel]

    if len(uu) > args.num_points:
        sel = np.random.choice(len(uu), args.num_points, replace=False)
        uu = uu[sel]
        vv = vv[sel]

    sid = seg_raw[vv, uu]
    if not args.include_background:
        keep = np.array([int(s) not in background_ids for s in sid], dtype=bool)
        uu = uu[keep]
        vv = vv[keep]
        sid = sid[keep]

    if len(uu) == 0:
        raise RuntimeError("No observer-camera points remain after background filtering")
    p_world0 = np.array([point_texture_to_world(source_camera, position, u, v) for u, v in zip(uu, vv)], dtype=np.float32)
    world_min = p_world0.min(axis=0)
    world_max = p_world0.max(axis=0)

    owner_sid = []
    local_points = []
    init_world = []
    for p, s in zip(p_world0, sid):
        entity = entity_map.get(int(s), None)
        if entity is None:
            continue
        pose = entity.get_pose().to_transformation_matrix().astype(np.float32)
        pose_inv = np.linalg.inv(pose)
        p_local = (pose_inv @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32))[:3]
        owner_sid.append(int(s))
        local_points.append(p_local)
        init_world.append(p)

    if len(local_points) == 0:
        raise RuntimeError("Failed to attach observer-view points to entities")

    local_points = np.asarray(local_points, dtype=np.float32)
    init_world = np.asarray(init_world, dtype=np.float32)

    markers, colors = create_markers(task.scene, len(local_points), radius=args.marker_radius)
    colorize_markers(markers, colors)

    original_update_render = task._update_render

    def wrapped_update_render():
        current_map = build_scene_entity_map(task)
        for i, marker in enumerate(markers):
            s = owner_sid[i]
            entity = current_map.get(s, None)
            if entity is not None:
                pose = entity.get_pose().to_transformation_matrix().astype(np.float32)
                p_world = (pose @ np.array([local_points[i, 0], local_points[i, 1], local_points[i, 2], 1.0], dtype=np.float32))[:3]
            else:
                p_world = init_world[i]
            marker.set_pose(sapien.Pose(p=p_world.tolist()))
        original_update_render()

    task._update_render = wrapped_update_render

    robot_count = sum(1 for s in owner_sid if s in robot_ids)
    object_count = len(owner_sid) - robot_count
    print(f"[ObserverReplay] episode={ep}, camera={args.camera}, sampled_points={len(owner_sid)}")
    print(f"[ObserverReplay] robot_points={robot_count}, object_points={object_count}")
    print(f"[ObserverReplay] world bounds min={world_min} max={world_max}")
    print("Close the SAPIEN viewer window to exit.")

    task.play_once()

    while True:
        try:
            task._update_render()
            task.viewer.render()
            task.scene.step()
        except Exception:
            break


if __name__ == "__main__":
    main()