# Initial 7-Class Training and Sampling Policy

Status: accepted initial configuration, implemented for both
`pyg_gravnet_v1` and `sparse_resnet_v1`.

## Frozen baseline

The first comparable GravNet and sparse 3D ResNet runs use the following
policy:

1. Train seven classes in this fixed order: `CC_numu`, `CC_nue`, `NC`,
   `neutral_kaon`, `neutron`, `passing_muon`, and `muon_DIS`.
2. Exclude `CC_nutau` completely.
3. Draw the seven training classes uniformly. With the configured 100,000
   sampled events per epoch, the expectation is about 14,286 events per class.
4. Within a selected class, draw data/MC domains in their natural proportion.
   Here “natural” means proportional to the number of eligible training events
   in that class and domain; no explicit domain reweighting is applied.
5. Within each selected Muon-DIS domain, draw uniformly over its populated
   density bins. Then draw an event naturally inside the chosen bin. Events in
   the `invalid` density category are not eligible for sampled training.
6. Keep validation natural: visit every event in the held-out fold once, with
   no class, domain, or density resampling. Muon-DIS events marked `invalid`
   remain in validation.
7. Optimize ordinary, unweighted multiclass cross-entropy.
8. Select the best checkpoint and perform early stopping by maximizing
   `val/macro_f1` (patience 10). Also inspect the natural-distribution loss,
   balanced accuracy, confusion matrix, and domain/density diagnostics.

This separates two concerns: balanced training exposure prevents the largest
classes and densest Muon-DIS bins from dominating gradient updates, while the
natural held-out fold measures performance on the population the model will
actually encounter. Class weights are deliberately not combined with uniform
class sampling in the baseline, because that would correct the same imbalance
twice.

## Configuration contract

Both active configuration files contain these values:

```yaml
data:
  epoch_size: 100000
  class_probabilities: uniform
  domain_probabilities: null
  muondis_density_probabilities: uniform
  validation_sampling: natural

objective:
  name: cross_entropy
  class_weights: null

checkpoint:
  monitor: val/macro_f1
  mode: max
```

`validate_scifi_classification_config` rejects departures from the class,
sampling, objective, and checkpoint-monitor contract, including if `CC_nutau`
is no longer the sole excluded class. `epoch_size`, maximum epochs, and early
stopping patience may still be reduced in copied smoke-test configs. The
hierarchical sampler implements all three training decisions explicitly. The
validation dataset independently enforces `sampling_strategy: natural` in the
fold view.

## Monitoring and later decisions

W&B project: `snd-ml-data-driven-3d`.

For every run, check the observed training population by class, domain, and
Muon-DIS density bin; the sampler is probabilistic, so counts will fluctuate
around their expectations. Compare models using natural validation macro-F1
and per-class recall, not training loss alone.

Do not change this baseline before the first controlled GravNet and sparse
ResNet comparison. If uniform density-bin sampling gives unstable gradients or
hurts common-bin performance, the next controlled experiment should change
only the Muon-DIS bin probability (for example, square-root-frequency
sampling), retaining natural validation and the unweighted loss.

## Verification

Focused tests cover:

- uniform versus natural Muon-DIS density-bin probabilities;
- natural validation and retention of invalid-density events;
- class/domain hierarchy and reproducibility;
- strict configuration validation for the frozen policy; and
- unweighted cross-entropy behavior.
