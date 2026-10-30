"""Mask out non-face regions and lightly smooth frames before v2e.

Designed to run frame by frame (causal), so the same FacePreprocessor can be
used on a live camera stream. For offline use it processes an image folder:

    python preprocess_face.py --clip s19_01

reads input/s19_01/*.jpg and writes input/s19_01_masked/*.png with the same
base names, ready to pass to v2e with -i input/s19_01_masked.
"""
import argparse
import glob
import os

import cv2
import numpy as np


class FacePreprocessor:
    def __init__(
        self,
        blur_sigma=1.0,
        temporal_alpha=0.6,
        ellipse_width=0.40,
        ellipse_height=0.40,
        ellipse_center_y=0.48,
        feather_px=6,
        redetect_every=50,
        move_tol_px=6,
    ):
        """
        blur_sigma: spatial Gaussian blur sigma in pixels (0 disables)
        temporal_alpha: weight of the new frame in a running average,
            1.0 disables temporal smoothing
        ellipse_width/height: ellipse semi-axes as a fraction of face box size
        ellipse_center_y: ellipse center as a fraction of face box height
        feather_px: soft edge width of the mask
        redetect_every: run face detection every N frames
        move_tol_px: only move the mask if the face moved more than this,
            since every mask move creates events along its edge
        """
        self.blur_sigma = blur_sigma
        self.temporal_alpha = temporal_alpha
        self.ellipse_width = ellipse_width
        self.ellipse_height = ellipse_height
        self.ellipse_center_y = ellipse_center_y
        self.feather_px = feather_px
        self.redetect_every = redetect_every
        self.move_tol_px = move_tol_px

        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.frame_idx = 0
        self.face_box = None
        self.mask = None
        self.fill_value = None
        self.smoothed = None

    def detect_face(self, gray):
        # Pad so tightly cropped faces that touch the border are still found
        pad = 40
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        min_side = int(min(gray.shape) * 0.4)
        faces = self.detector.detectMultiScale(
            padded, scaleFactor=1.05, minNeighbors=4, minSize=(min_side, min_side))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return x - pad, y - pad, w, h

    def build_mask(self, shape, box):
        h_img, w_img = shape
        x, y, w, h = box
        center = (int(x + w / 2), int(y + h * self.ellipse_center_y))
        axes = (int(w * self.ellipse_width), int(h * self.ellipse_height))

        mask = np.zeros((h_img, w_img), np.float32)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        if self.feather_px > 0:
            k = self.feather_px * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), self.feather_px / 2)
        return mask

    def update_face(self, gray):
        box = self.detect_face(gray)
        if box is None:
            if self.face_box is None:
                # No face yet: fall back to a centered box covering the frame
                h, w = gray.shape
                box = (0, 0, w, h)
            else:
                return

        if self.face_box is not None:
            old_cx = self.face_box[0] + self.face_box[2] / 2
            old_cy = self.face_box[1] + self.face_box[3] / 2
            new_cx = box[0] + box[2] / 2
            new_cy = box[1] + box[3] / 2
            if max(abs(new_cx - old_cx), abs(new_cy - old_cy)) <= self.move_tol_px:
                return

        self.face_box = box
        self.mask = self.build_mask(gray.shape, box)

    def process(self, frame):
        """Takes a BGR or gray uint8 frame, returns the processed gray uint8 frame."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self.mask is None or self.frame_idx % self.redetect_every == 0:
            self.update_face(gray)

        img = gray.astype(np.float32)

        if self.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)

        if self.smoothed is None or self.temporal_alpha >= 1.0:
            self.smoothed = img
        else:
            a = self.temporal_alpha
            self.smoothed = a * img + (1 - a) * self.smoothed

        # Fill the masked-out area with one constant gray so it never makes events
        if self.fill_value is None:
            self.fill_value = float(np.mean(self.smoothed[self.mask > 0.5]))

        out = self.mask * self.smoothed + (1 - self.mask) * self.fill_value

        self.frame_idx += 1
        return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--temporal_alpha", type=float, default=0.6)
    args = parser.parse_args()

    in_dir = f"input/{args.clip}"
    out_dir = f"input/{args.clip}_masked"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(f"{in_dir}/*.*"))
    if len(files) == 0:
        raise RuntimeError(f"No images found in {in_dir}")

    pre = FacePreprocessor(
        blur_sigma=args.blur_sigma, temporal_alpha=args.temporal_alpha)

    for f in files:
        out = pre.process(cv2.imread(f))
        # PNG so we don't add new JPEG compression noise
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, name), out)

    print(f"Wrote {len(files)} frames to {out_dir}")
    print(f"Face box: {pre.face_box}")


if __name__ == "__main__":
    main()
