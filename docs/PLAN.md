# Plan

Task: xis.ai "Measure It" v2.0. Photograph an object, report real width and height in mm.
Timeline note: assessment issued 29 Oct 2026. I worked in sessions; the dates of each commit in git show the real order. Anything not finished is listed in DEFINITION_OF_DONE.md.

## Object and reference
- Object: a rigid, flat rectangular object (box or hardcover notebook, about 15 x 10 cm). Chosen because it is rigid, flat, and has clear edges, so the pixel-to-mm homography is valid.
- Reference: printed ArUco marker (DICT_4X4_50, ID 0), printed at 70 mm, flat in the same plane as the object. Its black-square side is measured by me with a ruler (value in README).

## Order and hours (8h total)
Setup 0.5, calibration 1, photos and labelling 2, training 1 (background, not counted), inference 1.5, accuracy and undistortion experiment 1, tilt and refusal 0.5, docs and Loom 0.5.

## Model choice
torchvision Mask R-CNN (ResNet50-FPN, COCO-pretrained), fine-tuned on my own images, 2 classes (background, object). Chosen because it gives instance masks and boxes in one model, has a pretrained backbone that works with ~70 images, and is not Ultralytics YOLO or Roboflow.

## Split
Seed 42, 70/20/10, files in dataset/splits/, committed before any code. Test is never used for training, tuning or checkpoint choice. Checkpoint is chosen on val only.

## Pixel to mm
Undistort with camera.npz, find marker corners, cv2.getPerspectiveTransform from marker corners to a square of known mm, transform the mask contour, cv2.minAreaRect gives width and height. Valid only if object and marker are on the same flat plane and the image is undistorted.

## Skipping
ChArUco (used checkerboard), lighting and distance experiments (only tilt), a polished UI unless time is left.

## Decision record
(filled at the end: approach taken, approach rejected, what rejecting it cost)
