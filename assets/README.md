# Go2 experiment assets

This directory contains the complete robot resources used by the experiment:

- `robots/go2/mjcf/scene.xml`: canonical MuJoCo scene and flat terrain.
- `robots/go2/mjcf/go2.xml`: Go2 MJCF model.
- `robots/go2/mjcf/meshes/`: every mesh referenced by the MJCF model.
- `robots/go2/usd/go2.usd`: Isaac Lab robot model.
- `robots/go2/usd/Props/instanceable_meshes.usd`: geometry referenced by the USD model.

The Isaac Lab rough terrain is generated procedurally at runtime by
`train/isaac/pytorch/setup.py`. It does not
depend on a terrain mesh or another external asset. The MuJoCo transfer stage
uses the flat plane defined directly in `scene.xml`, matching the main-branch
experiment preset.
