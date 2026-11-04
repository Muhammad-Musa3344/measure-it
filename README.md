# Measure It

Photograph an object and get its real width and height in millimetres.
Pipeline: calibrate camera, undistort, segment, convert pixels to mm with a reference marker, validate against a calliper.

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## How to run

Calibrate:   python calibration/calibrate.py
Measure:     python inference/measure.py <image>

(Each stage is added as it is built.)

## Hardware

- CPU / GPU / RAM / OS: <fill in>
- Camera: <phone model>, <resolution>, focus and zoom locked

## Dataset

- Object: <name, approx size, why chosen>
- Images: <count>, split 70 / 20 / 10 (see dataset/splits/)
- Labelling tool: <CVAT / Label Studio>

## Model weights

- Link: <release URL>
- SHA-256: <hash>