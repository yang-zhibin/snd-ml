# Data-Driven 3D-Hit CNN Plan

## 1. Define and prepare the dataset

- Define four physics classes: neutrino, neutral hadron, muon-like, and muon-DIS-like.
- Use the currently available Hit3D events; completing every production is not required.
- Match Hit3D events with feature metadata, particularly `density_sndsw_scifi`.
- Develop the muon-DIS sampling scheme across the SciFi-density range.
- Balance the physics classes and the available data/MC representations for training.
- Assign five source-grouped folds while preserving class, domain, SciFi-density, neutral-hadron energy, and run coverage.
- Keep training weights separate from luminosity and physics weights.

The common dataset manifest should contain one row per event with, at minimum:

`event_id, hit3d_path, class, subclass, domain, source_group, fold, density_sndsw_scifi, energy_bin, training_weight, physics_weight`

## 2. Build the 3D-hit CNN

- Implement a classifier using the current sparse Hit3D format.
- Train one model for each of the five held-out folds.
- Save class scores and event metadata for all out-of-fold events.
- Establish a simple baseline before considering domain adaptation or more complex architectures.

## 3. Evaluate the model and produce the physics result

- Combine out-of-fold predictions so that every event is evaluated exactly once.
- Compare performance between folds, runs, data and MC domains, SciFi-density intervals, and neutral-hadron energy bins.
- Derive selection efficiencies and uncertainty estimates from these comparisons.
- Produce final score distributions and yields using observed data and luminosity-normalized MC.
- Use training weights only during optimization, not for reported physics yields.

## 4. Perform geometry-robustness tests

- Translate events in the x-y plane without retraining the nominal model.
- Test contained translations and cropped edge translations separately.
- Measure score changes, class migration, and efficiency versus displacement and lost activity.
- Use the results to decide whether geometry augmentation, recentering, or an additional systematic uncertainty is needed.

Detailed choices such as SciFi-density bin boundaries, sample sizes, weight caps, CNN architecture, and selection thresholds will be decided when each stage is implemented.
