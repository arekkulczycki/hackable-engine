from gym.envs.registration import register


register(
    id='chess-v0',
    entry_point='arek_chess.training.envs:FullBoardEnv',
)
