"""
Rigid-pose-driven point tracking visualization (no ICP)

Pipeline:
1. Sample points from frame-0 actor segmentation mask.
2. Back-project sampled pixels to world points.
3. Convert frame-0 world points into the selected rigid actor local frame.
4. For each keyframe, transform local points by scene_state/rigid_actor_poses.
5. Project to image and render tracks.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


class RigidPosePointTracking:
    def __init__(self, hdf5_path: str, camera_name: str = "head_camera"):
        self.hdf5_path = hdf5_path
        self.camera_name = camera_name
        self.h5 = h5py.File(hdf5_path, "r")
        self.num_frames = self.h5[f"observation/{camera_name}/depth"].shape[0]

    def get_camera_intrinsic(self, frame_idx: int) -> np.ndarray:
        return self.h5[f"observation/{self.camera_name}/intrinsic_cv"][frame_idx].astype(np.float32)

    def get_cam2world(self, frame_idx: int) -> np.ndarray:
        return self.h5[f"observation/{self.camera_name}/cam2world_gl"][frame_idx].astype(np.float32)

    def get_depth(self, frame_idx: int) -> np.ndarray:
        depth = self.h5[f"observation/{self.camera_name}/depth"][frame_idx].astype(np.float32)
        depth[depth <= 0] = np.nan
        return depth

    def get_actor_seg_raw(self, frame_idx: int) -> np.ndarray:
        path = f"observation/{self.camera_name}/actor_segmentation_raw"
        if path in self.h5:
            return self.h5[path][frame_idx].astype(np.int32)
        # fallback: color segmentation first channel (legacy)
        seg = self.h5[f"observation/{self.camera_name}/actor_segmentation"][frame_idx]
        return seg[..., 0].astype(np.int32)

    def get_rgb(self, frame_idx: int) -> np.ndarray:
        rgb = self.h5[f"observation/{self.camera_name}/rgb"][frame_idx]
        if isinstance(rgb, (bytes, bytearray)):
            rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
        elif hasattr(rgb, "dtype") and rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        return rgb

    def get_rigid_actor_pose(self, actor_name: str, frame_idx: int) -> np.ndarray:
        path = f"scene_state/rigid_actor_poses/{actor_name}"
        return self.h5[path][frame_idx].astype(np.float32)

    def choose_target_actor(self) -> str:
        keys = list(self.h5["scene_state/rigid_actor_poses"].keys())
        candidates = [k for k in keys if k not in {"ground", "table", "wall"}]
        if not candidates:
            raise ValueError("No movable rigid actor found in scene_state/rigid_actor_poses")
        return candidates[0]

    def depth_pixels_to_world(self, u: np.ndarray, v: np.ndarray, d: np.ndarray, frame_idx: int) -> np.ndarray:
        K = self.get_camera_intrinsic(frame_idx)
        T_c2w = self.get_cam2world(frame_idx)

        x = (u - K[0, 2]) * d / K[0, 0]
        y = (v - K[1, 2]) * d / K[1, 1]
        z = d
        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
        points_world = (T_c2w @ points_cam.T).T[:, :3]
        return points_world.astype(np.float32)

    def world_to_pixels(self, points_world: np.ndarray, frame_idx: int):
        K = self.get_camera_intrinsic(frame_idx)
        T_w2c = np.linalg.inv(self.get_cam2world(frame_idx))
        pts_h = np.hstack([points_world, np.ones((len(points_world), 1), dtype=np.float32)])
        pts_cam = (T_w2c @ pts_h.T).T[:, :3]

        eps = 1e-6
        z = np.maximum(pts_cam[:, 2], eps)
        uv_h = (K @ pts_cam.T).T
        uv = uv_h[:, :2] / z[:, None]

        H, W = self.get_depth(frame_idx).shape
        in_view = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pts_cam[:, 2] > 0)
        return uv.astype(np.int32), in_view

    def run(self, output_dir: str, num_keyframes: int = 8, num_points: int = 600):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        actor_name = self.choose_target_actor()
        print(f"Target rigid actor: {actor_name}")

        frame0 = 0
        depth0 = self.get_depth(frame0)
        seg0 = self.get_actor_seg_raw(frame0)

        valid = (~np.isnan(depth0)) & (depth0 > 0) & (seg0 > 0)
        if valid.sum() == 0:
            raise ValueError("No valid segmented pixels in frame 0")

        ids, counts = np.unique(seg0[valid], return_counts=True)
        target_seg_id = int(ids[np.argmax(counts)])
        print(f"Target segmentation id: {target_seg_id} (pixels={int(counts.max())})")

        mask = valid & (seg0 == target_seg_id)
        vv, uu = np.where(mask)
        if len(uu) == 0:
            raise ValueError("Target segmentation id has no valid depth pixels")

        if len(uu) > num_points:
            pick = np.random.choice(len(uu), num_points, replace=False)
            uu = uu[pick]
            vv = vv[pick]

        d0 = depth0[vv, uu]
        points_world0 = self.depth_pixels_to_world(uu.astype(np.float32), vv.astype(np.float32), d0, frame0)

        pose0 = self.get_rigid_actor_pose(actor_name, frame0)
        pose0_inv = np.linalg.inv(pose0)
        p0_h = np.hstack([points_world0, np.ones((len(points_world0), 1), dtype=np.float32)])
        points_local = (pose0_inv @ p0_h.T).T[:, :3]

        keyframes = np.linspace(0, self.num_frames - 1, num_keyframes, dtype=int)
        print(f"Keyframes: {keyframes.tolist()}")

        colors = []
        n = len(points_local)
        for i in range(n):
            ratio = i / max(1, n - 1)
            h = int(240 * (1 - ratio))
            c = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            colors.append((int(c[0]), int(c[1]), int(c[2])))

        frames = []
        prev_uv, prev_valid = None, None

        for fidx in keyframes:
            rgb = self.get_rgb(int(fidx)).copy()

            pose_t = self.get_rigid_actor_pose(actor_name, int(fidx))
            local_h = np.hstack([points_local, np.ones((len(points_local), 1), dtype=np.float32)])
            points_world_t = (pose_t @ local_h.T).T[:, :3]

            uv, valid_t = self.world_to_pixels(points_world_t, int(fidx))

            if prev_uv is not None:
                step = max(1, len(uv) // 60)
                for i in range(0, len(uv), step):
                    if valid_t[i] and prev_valid[i]:
                        cv2.line(rgb, tuple(prev_uv[i]), tuple(uv[i]), colors[i], 2)

            for i, (p, ok) in enumerate(zip(uv, valid_t)):
                if ok:
                    cv2.circle(rgb, tuple(p), 3, colors[i], -1)

            cv2.putText(rgb, f"Frame {int(fidx)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(rgb, f"Points tracked: {int(valid_t.sum())}/{len(valid_t)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(rgb, f"Actor: {actor_name}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imwrite(str(out / f"frame_{int(fidx):03d}.png"), rgb)
            frames.append(rgb)
            prev_uv, prev_valid = uv, valid_t

        gif_path = out / "point_tracking.gif"
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=300, loop=0)

        mp4_path = out / "point_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(mp4_path), fourcc, 3.33, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            vw.write(f)
        vw.release()

        print(f"Saved visualization to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Rigid-pose point tracking (no ICP)")
    parser.add_argument("hdf5_path", help="Path to episode hdf5")
    parser.add_argument("--output", "-o", default="point_tracking_v2", help="Output directory")
    parser.add_argument("--num-keyframes", type=int, default=8, help="Number of keyframes")
    parser.add_argument("--num-points", type=int, default=600, help="Number of sampled points")
    parser.add_argument("--camera", default="head_camera", help="Camera name")
    args = parser.parse_args()

    tracker = RigidPosePointTracking(args.hdf5_path, camera_name=args.camera)
    tracker.run(output_dir=args.output, num_keyframes=args.num_keyframes, num_points=args.num_points)


if __name__ == "__main__":
    main()
