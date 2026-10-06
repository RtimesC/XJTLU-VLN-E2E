# Paper 1 initial HM3D findings

These are mechanism-protocol findings, not original R2R/R4R/REVERIE results.
The runs use the fixed low-view vehicle contract and the explicit
`geometry` semantic backend.

## Evidence collected

| Scene | Steps | Gaussian count | Collisions |
|---|---:|---:|---:|
| `TEEsavR23oF` | 30 | 2785 | 0 |
| `HaxA7YrQdEC` | 30 | 326 | 9 |
| `wcojb4TFT35` | 30 | 2090 | 0 |

The same pipeline and camera contract were used for all three scenes. The
records are stored on the Linux evaluation host under
`artifacts/paper1_hm3d/multiseed/` and include the commit, trajectory, actions,
map configuration, and failure tags.

## Interpretation

- RGB-D back-projection and incremental Gaussian accumulation are operational.
- Map growth is strongly scene-dependent; a low primitive count is a useful
  signal for coverage or observability failure, not a navigation score.
- One scene generated repeated collision blocking while the other two did not.
  This makes embodiment-aware local geometry and recovery a concrete control
  stress case.
- All runs still use the explicit geometry-only semantic backend. Open-set
  instance and language grounding have not been measured yet.

## Current engineering entry point

The first candidate is:

> **Uncertainty-aware open-set semantic Gaussian mapping for a 0.45 m
> limited-FOV agent.**

The proposed contribution is to estimate map confidence from observation count,
depth stability, viewpoint coverage, and semantic agreement, then use that
uncertainty to decide whether to continue, rotate for an informative view, or
recover from a blocked command. This is testable in the current Habitat setup
and directly connects the paper-1 OSG idea to the project's low-view vehicle
constraint.

This remains a candidate until the SAM2+CLIP backend is available and the same
three-scene ablations compare geometry-only, semantic grouping, and uncertainty
aware policies.
