# Region V1 Veto edge-muon selection

Status: proposed selection definition; not yet implemented or evaluated  
Recorded: 2026-10-02  
Analysis: Region V1 ABCD background estimation

## Purpose

Define a Veto-position selection that **keeps edge-muon candidates**. Missing
measurements are neutral: if one detector orientation is unavailable, the
other orientation makes the decision. An event with no valid position in any
Veto plane does not pass the edge-muon selection.

This note freezes the requested logical definition only. It does not authorize
implementation, production processing, threshold optimization, or a change to
the existing Region V1 baseline.

## Geometry and edge boundaries

The boundaries below were extracted from the installed bar surfaces in the
frozen 2024 geometry file:

```text
/eos/experiment/sndlhc/convertedData/physics/2024/
geofile_sndlhc_TI18_V12_2024.root
```

| Veto plane | Measured coordinate | Active minimum [cm] | Active maximum [cm] | Low-edge threshold [cm] | High-edge threshold [cm] |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | y | 14.835139 | 56.867571 | 24.835139 | 46.867571 |
| 2 | y | 12.784356 | 54.816788 | 22.784356 | 44.816788 |
| 3 | x | -47.370290 | -5.755074 | -37.370290 | -15.755074 |

Each edge band extends 10 cm inward from the corresponding physical detector
boundary. These are nominal geometry boundaries. A run-dependent active-area
definition would additionally need to account for dead or masked channels.

## Valid-position definitions

The feature-tree missing-position sentinel is protected against explicitly.
A plane position is valid only when the plane contains at least one hit and
its average coordinate is not the missing value:

```python
valid_veto1 = count_veto1 > 0 and avg_veto1_y > -900
valid_veto2 = count_veto2 > 0 and avg_veto2_y > -900
valid_veto3 = count_veto3 > 0 and avg_veto3_x > -900
```

## Frozen selection logic

```python
v1_low  = valid_veto1 and avg_veto1_y <= 24.835139
v1_high = valid_veto1 and avg_veto1_y >= 46.867571

v2_low  = valid_veto2 and avg_veto2_y <= 22.784356
v2_high = valid_veto2 and avg_veto2_y >= 44.816788

any_y_valid = valid_veto1 or valid_veto2
any_valid = any_y_valid or valid_veto3

if valid_veto1 and valid_veto2:
    # With both horizontal planes present, require the same physical y edge.
    y_edge = ((v1_low and v2_low) or
              (v1_high and v2_high))
elif valid_veto1:
    y_edge = v1_low or v1_high
elif valid_veto2:
    y_edge = v2_low or v2_high
else:
    # Missing y is neutral; let the x orientation decide.
    y_edge = True

if valid_veto3:
    x_edge = (
        avg_veto3_x <= -37.370290
        or avg_veto3_x >= -15.755074
    )
else:
    # Missing x is neutral; let the y orientation decide.
    x_edge = True

edge_muon = any_valid and y_edge and x_edge

# This selection retains edge-muon candidates.
keep_event = edge_muon
```

The `any_valid` guard is required. Without it, an event missing every Veto
position would have both neutral orientation flags set to `True` and would be
incorrectly classified as an edge muon.

## Decision table

| Available position information | Edge-muon decision |
| --- | --- |
| Both y planes and x | Both orientations must identify an edge; the two y planes must identify the same y edge |
| One y plane and x | The available y plane and x must both identify an edge |
| y only | The available y plane or consistent two-plane y result decides |
| x only | Veto plane 3 decides |
| No valid Veto position | `edge_muon = False`; the event is not retained by this selection |

Here, an unavailable orientation is neutral, while an available position in
the central detector area is a real failure of the edge condition. The
complementary non-edge sample, if needed for another study, is
`not edge_muon`; it is not the `keep_event` population defined here.

## Required checks before use

Before treating this as a physics selection, report the accepted counts and
efficiencies separately for events with both orientations, y only, x only,
and no valid Veto position. In particular, the one-plane fallbacks can be more
sensitive to isolated or noisy hits than the same-edge two-plane requirement.
The thresholds must not be optimized using nominal signal-region data.
