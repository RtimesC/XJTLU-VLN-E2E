# Paper 1 HM3D protocol

This repository implements a mechanism-level protocol inspired by *3D
Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation*.
It is not a numerical reproduction of the R2R, R4R, or REVERIE results.

## Pipeline

```text
RGB-D + camera pose -> ESM GaussianMap -> OSG backend -> MAP predictor
                                               -> discrete or continuous action
```

The first runnable backend is `geometry`. It verifies RGB-D back-projection,
incremental voxel-merged Gaussian accumulation, and the episode/action record
schema. `sam2clip` is an explicit option, but currently fails fast until
checkpoint-backed SAM2 and CLIP implementations are installed on the Linux
evaluation environment.

## Linux run

```bash
cd ~/Desktop/XJTLU-VLN-E2E
git pull --ff-only origin spatial-intelligence-reasoning
SCENE=/home/sousuke/Desktop/habitat-lab/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.glb
/home/sousuke/miniforge3/envs/habitat_vln/bin/python scripts/run_paper1_hm3d_protocol.py \
  --scene "$SCENE" --steps 50 --seed 0 --action-mode discrete \
  --semantic-backend geometry --output artifacts/paper1_hm3d/discrete.jsonl
```

Use `--action-mode continuous` for the `(v, omega, stop)` transfer line. The
record stores commit, camera/agent/map/policy configuration, raw/safe/executed
actions, trajectory, map size, collisions, and failure tags.

## Current evidence boundary

- The runner uses the fixed 0.45 m, 640x480, 90 degree RGB-D camera contract.
- HM3D navmesh is loaded or recomputed at runtime with the vehicle radius and
  height.
- Geometry-only runs are protocol smoke tests, not open-set semantic results.
- Spot is an embodied visual carrier; full legged dynamics remain out of scope.
