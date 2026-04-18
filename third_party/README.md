# Vendored Third-Party Components

This repository vendors a minimal set of third-party code required for the
released `DBP pretrain` workflows.

## Included Components

- `metaworld`
  - Purpose: MetaWorld runtime required by Adroit / MetaWorld dataset
    generation.
- `mj_envs`
  - Purpose: Adroit environment registration required by Adroit dataset
    generation.
- `adroit_metaworld_support`
  - Purpose: Adroit / MetaWorld point-cloud data generation used by DBPO.

Only the minimal subset required for dataset generation is vendored here.
Training code outside the released DBPO workflow and unrelated benchmark
utilities are not part of the default installation surface.
