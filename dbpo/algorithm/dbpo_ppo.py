"""Stage-1 placeholder for the stage-2 DBPO PPO adapter."""


def __getattr__(name: str):
    raise ImportError(
        "DBPO online finetuning is not part of stage 1. "
        "Do not import dbpo.algorithm.dbpo_ppo in the stage-1 repository."
    )
