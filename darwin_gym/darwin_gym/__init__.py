from gym.envs.registration import register

register(
    id='pybullet:DarwinBulletEnv-v0',
    entry_point='darwin_gym.envs:DarwinBulletEnv',
)
