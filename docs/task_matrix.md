# Task Matrix

This matrix defines the released public support boundary for:

- stage 1: `DBP pretrain`
- stage 2: `DBPO gym finetune`

## Native Simulated Manipulation Tasks

Supported:

- `pusht_lowdim`
- `pusht_image`
- `blockpush_lowdim_seed`
- `blockpush_lowdim_seed_abs`
- `kitchen_lowdim`
- `kitchen_lowdim_abs`
- `can_lowdim`
- `can_lowdim_abs`
- `can_image`
- `can_image_abs`
- `lift_lowdim`
- `lift_lowdim_abs`
- `lift_image`
- `lift_image_abs`
- `square_lowdim`
- `square_lowdim_abs`
- `square_image`
- `square_image_abs`
- `tool_hang_lowdim`
- `tool_hang_lowdim_abs`
- `tool_hang_image`
- `tool_hang_image_abs`
- `transport_lowdim`
- `transport_lowdim_abs`
- `transport_image`
- `transport_image_abs`

## Adroit / MetaWorld Tasks

Adroit:

- `adroit_door`
- `adroit_hammer`
- `adroit_pen`

MetaWorld:

- all released `metaworld_*` task configs in `configs/task/`

## Gym Tasks

Public gym tasks:

- `hopper-medium-v2`
- `ant-medium-expert-v2`
- `walker2d-medium-v2`

These released tasks are included in `DBP pretrain` and use the unified
checkpoint schema required by stage 2.

## Support Notes

- Stage 1 exposes `DBP pretrain` across the three task families above.
- Stage 2 exposes `DBPO gym finetune` for:
  - `hopper-medium-v2`
  - `ant-medium-expert-v2`
  - `walker2d-medium-v2`
- `PushT`, `BlockPush`, and `Kitchen` runtime behavior still requires complete
  server-side validation.
- Point-cloud tasks have released dataset and policy contracts, but real data
  generation remains server-side work.
