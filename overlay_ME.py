import cv2
import numpy as np
import math
import argparse
import os
import ast


parser = argparse.ArgumentParser()
parser.add_argument("--clip", required=True)
parser.add_argument("--onset", type=float, required=True)
parser.add_argument("--offset", type=float, required=True)
args = parser.parse_args()

clip = args.clip
onset_phys = args.onset
offset_phys = args.offset

run_dir = f"output/{clip}_me"
orig_path = os.path.join(run_dir, "video_orig.avi")
dvs_path = os.path.join(run_dir, "dvs-video.avi")
out_path = os.path.join(run_dir, "overlay_events_onset_offset.mp4")


output_fps = 30.0
slowdown = 4.0

gray_center = 127
threshold = 18
alpha = 0.85

pos_color = np.array([0, 255, 0], dtype=np.float32)
neg_color = np.array([0, 0, 255], dtype=np.float32)

def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def read_v2e_args(run_dir):
    # Read the settings v2e saved in v2e-args.txt into a dict.
    settings = {}
    with open(os.path.join(run_dir, "v2e-args.txt")) as f:
        for line in f:
            key,sep, value = line.rstrip("\n").partition(":\t")
            if not sep:
                continue
            try:
                settings[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                settings[key] = value
    return settings

def read_dvs_frame_times(run_dir):
    #Time in seconds of each frame in dvs-video.avi (middle of its window).
    path = os.path.join(run_dir, "dvs-video-frame_times.txt")
    return np.loadtxt(path, comments="#", usecols=1)


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
    )
    cap.release()
    return info


    
orig_frames = read_all_frames(orig_path)
dvs_frames = read_all_frames(dvs_path)
    
if len(orig_frames) == 0:
    raise RuntimeError(f"No frames read from {orig_path}")

if len(dvs_frames) == 0:
    raise RuntimeError(f"No frames read from {dvs_path}")

v2e_args = read_v2e_args(run_dir)

orig_physical_fps = v2e_args.get("input_frame_rate")
if orig_physical_fps is None:
    cap = cv2.VideoCapture(orig_path)
    orig_physical_fps = cap.get(cv2.CAP_PROP_FPS) * v2e_args.get("input_slowmotion_factor", 1.0)
    cap.release()
orig_physical_fps = float(orig_physical_fps)

dvs_times = read_dvs_frame_times(run_dir)
if len(dvs_times) != len(dvs_frames):
    print(f"Warning: {len(dvs_times)} frame times but {len(dvs_frames)} DVS frames, using the shorter")
    n = min(len(dvs_times), len(dvs_frames))
    dvs_times, dvs_frames = dvs_times[:n], dvs_frames[:n]
dvs_physical_fps = 1.0 / np.median(np.diff(dvs_times))

print(f"Original Video: {orig_physical_fps:.1f} fps (from v2e-args.txt)")
print(f"DVS Video: {dvs_physical_fps:.1f} fps (from dvs-video-frame_times.txt)")

h,w = orig_frames[0].shape[:2]

if dvs_frames[0].shape[:2] != (h,w):
    dvs_frames = [
        cv2.resize(frame, (w,h), interpolation=cv2.INTER_NEAREST)
        for frame in dvs_frames
    ]

duration_phys = len(orig_frames) / orig_physical_fps

duration_out = duration_phys * slowdown
n_out_frames = int(math.ceil(duration_out * output_fps))

onset_out = onset_phys * slowdown
offset_out = offset_phys * slowdown

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, output_fps, (w,h))

for out_idx in range(n_out_frames):
    t_out = out_idx / output_fps
    t_phys = t_out / slowdown
    idx_orig = min(int(round(t_phys * orig_physical_fps)), len(orig_frames) - 1)
    idx_dvs = int(np.argmin(np.abs(dvs_times - t_phys)))

    base = orig_frames[idx_orig].copy()

    if len(base.shape) == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    dvs = dvs_frames[idx_dvs]

    if len(dvs.shape) == 3:
        dvs_gray = cv2.cvtColor(dvs, cv2.COLOR_BGR2GRAY)
    else:
        dvs_gray = dvs
    delta = dvs_gray.astype(np.int16) - gray_center

    pos_mask = delta > threshold
    neg_mask = delta < -threshold

    overlay = base.astype(np.float32)

    if np.any(pos_mask):
        overlay[pos_mask] = (1 - alpha) * overlay[pos_mask] + alpha * pos_color

    if np.any(neg_mask):
        overlay[neg_mask] = (1 - alpha) * overlay[neg_mask] + alpha * neg_color

    frame_out = np.clip(overlay, 0, 255).astype(np.uint8)

    if onset_out <= t_out <= onset_out + 0.8:
        cv2.putText(
            frame_out,
            "ME ONSET",
            (int(w* 0.25), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2,
            cv2.LINE_AA,
        )

    if offset_out <= t_out <= offset_out + 0.8:
        cv2.putText(
            frame_out,
            "ME OFFSET",
            (int (w * 0.25), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    writer.write(frame_out)
writer.release()


print()
print(f"Saved: {out_path}")
print(f"Physical clip duration: {duration_phys:.2f} s")
print(f"Output duration: {duration_out:.2f} s")
print(f"ME onset physical: {onset_phys:.2f} s")
print(f"ME offset physical: {offset_phys:.2f} s")
print(f"ME onset playback: {onset_out:.2f} s")
print(f"ME offset playback: {offset_out:.2f} s")