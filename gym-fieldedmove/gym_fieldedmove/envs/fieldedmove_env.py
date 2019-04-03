from random import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class FieldedMove(gym.Env):
	metadata = {'render.modes': ['console', 'human'],
				'video.frames_per_second' : 60}
	red = (255, 0, 0)
	blue = (0, 0, 255)

	def __init__(self):
		self.window_height = 84
		self.window_width = 84
		self.env_height = 84
		self.env_width = 84
		self.object_size = 3
		self.rad = 30

		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.env_height, self.env_width, 3), dtype=np.uint8)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
		self.position = np.zeros(2)
		self.velocity = np.zeros(2)
		self.acceleration = np.zeros(2)
		self.count = 0
		self.count_max = 100
		self.done = 0
		self.reward = 0

		# self.goal = np.ones(2) # temp
		self.goal_degree = 2*np.pi
		self.goal = [self.rad * np.sin(self.goal_degree), self.rad * np.cos(self.goal_degree)]
		self.magnitude = 15 #np.array([15, 15])
		self.field = np.array([1., 1.])
		self.period = 1 # temp


	def _update_state(self):
		self.state = np.zeros((self.env_height, self.env_width, 3), dtype=np.uint8)
		posX, posY = np.round(self.position).astype(int)
		self.state[posX:posX+self.object_size, posY:posY+self.object_size] = self.red
		posX_g, posY_g = np.round(self.goal).astype(int)
		self.state[posX_g:posX_g+self.object_size, posY_g:posY_g+self.object_size] = self.blue
		return self.state

	def calculate_fieldforce(self, velocity):
		x, y = velocity
		phi = np.arctan(y/(x+0.000001))
		F = np.array([np.sin(self.period * phi), -np.cos(self.period * phi)])
		mag_curr = self.magnitude * np.sqrt(x**2 + y**2)
		F = np.multiply(F, mag_curr)
		return F


	def step(self, target):
		if self.done == 1:
			print("Game Over")
			return [self._update_state(), self.reward, self.done, None]
		else:
			self.count += 1
			self.acceleration += target
			self.acceleration += self.calculate_fieldforce(self.velocity)
			self.velocity += self.acceleration # temp
			self.position += self.velocity
			dist_goal = np.sqrt(((self.position - self.goal) ** 2).sum())

			self.reward = 1 - dist_goal/self.rad # temp

			if dist_goal < 0.1:
				self.done = 1

			self.render()


		return [self._update_state(), self.reward, self.done, None]

	def reset(self):
		self.count = 0
		self.done = 0
		self.position = np.zeros(2)
		self.velocity = np.zeros(2)
		self.acceleration = np.zeros(2)
		self.goal_degree = random.randint(36) * (2*np.pi/36)
		self.goal = [self.rad*np.sin(self.goal_degree), self.rad*np.cos(self.goal_degree)]
		self.reward = 0
		return self.state

	def render(self, mode='console', close=False):
		if mode == 'console':
			print(self._update_state)

		elif mode == "human":
			try:
				import pygame
				from pygame import gfxdraw
			except ImportError as e:
				raise error.DependencyNotInstalled(
					"{}. (HINT: install pygame using `pip install pygame`".format(e))
			if close:
				pygame.quit()

if __name__ == "__main__":
	env = FieldedMove()
	for i in range(30):
		action = np.random.normal(0, 1, 2)
		state, reward, done, info = env.step(action)