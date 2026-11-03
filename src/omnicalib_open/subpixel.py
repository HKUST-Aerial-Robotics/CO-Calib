"""Refine network coordinates on the original image before frame selection."""
from __future__ import annotations

import cv2
import numpy as np


SUBPIXEL = {"method": "cornerSubPix", "win": 5, "max_iter": 30, "eps": 0.001,
            "coordinate_space": "original_image"}


def refine_points(image_bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    win = SUBPIXEL["win"]
    eligible = (np.isfinite(points).all(axis=1)
                & (points[:, 0] >= win) & (points[:, 0] < width - win)
                & (points[:, 1] >= win) & (points[:, 1] < height - win))
    if eligible.any():
        corners = points[eligible].reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(
            gray, corners, (win, win), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
             SUBPIXEL["max_iter"], SUBPIXEL["eps"]),
        )
        points[eligible] = corners.reshape(-1, 2)
    return points
