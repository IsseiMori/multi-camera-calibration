# multi-camera-calibration

## Usage
```
python calibrate.py [-W WIDTH] [-H HEIGHT]
                    [--grid_length] data_path

                    -W: Number of rows - 1 
                    -H: Number of columns - 1 
                    --grid_length: Length of a single grid
                    data_path: Path to the directory containing images and intrinsics.json
```

`-W` and `-H` are the number of rows - 1 and columns - 1 of the chessboard. This is because only the inner grids are used for detection.

## Data Format
Place the captured images in a folder. Each image must be named as `{camera_id}.png`, where `camera_id` is a unique id for each camera. 

Create `instrinsics.json` which contains a list of camera IDs and their camera intrinsics as follows.

```
{
    camera_id_1: [fx, fy, ppx, ppy],
    camera_id_2: [fx, fy, ppx, ppy]
}
```

`camera_id` is a unique string for each camera. `fx` and `fy` are the focal legthm and `ppx` and `ppy` are the pixel coordinates of the principal point (center of projection).