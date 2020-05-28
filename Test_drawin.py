import gym
import darwin_gym
import pybullet, pybullet_envs
env=gym.make("DarwinBulletEnv-v0",render=True)
obs=env.reset()

while(1):
  env.render(mode="human")
  env.step(env.action_space.sample())
