import os
import sys
import random
import time

import pygame
from pygame import gfxdraw
import pygame.locals

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# FPS = 10
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class FieldedMove(gym.Env):
	metadata = {'render.modes': ['console', 'human'],
				'video.frames_per_second' : 10}

	def __init__(self):
		self.window_height = 84
		self.window_width = 84
		self.env_height = 84
		self.env_width = 84
		self.object_size = 7
		self.rad = 25
		self.n_max_trial = 10000
		self.n_trial = None

		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.env_height, self.env_width, 3), dtype=np.uint8)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
		self.center = None #np.array([self.window_height/2., self.window_width/2.])
		self.position = None #np.array([self.window_height/2., self.window_width/2.])
		self.velocity = None #np.zeros(2)
		# self.acceleration = np.zeros(2)
		self.step_multiplier = 3

		self.count = None
		self.count_max = 100
		self.done = 0
		self.value = None
		self.reward = None
		self.rew_per_step = -0.5
		self.rew_multiplier = 1

		# self.goal = np.ones(2) # temp
		self.dist_goal_thres = 0.5
		self.goal_degree = None #2*np.pi
		self.goal = None #np.array([self.rad * np.sin(self.goal_degree), self.rad * np.cos(self.goal_degree)])
		# self.goal += self.center

		self.magnitude = 0.5 #np.array([15, 15])
		# self.field = np.array([1., 1.])
		self.period = None #random.randint(1,3)

		self.screen = None #pygame.display.set_mode((self.window_height, self.window_width))
		self.player_render = None
		self.goal_render = None
		self.clock = pygame.time.Clock()

		self.mode = 'none' # default mode
		self.reset()

	def set_seed(self, seed):
		np.random.seed(seed)
		random.seed(seed)

	def reset(self):
		self.count = 0
		self.done = 0
		self.n_trial = 0

		#self.period = random.randint(1, 4)
		self.period_x = 4 * random.random()
		self.period_y = 4 * random.random()
		# self.period_y = self.period_x

		self._reset_task()
		# if self.mode == "human":
			# self.player_render = None  # Object(BLUE, self.object_size, self.position[0], self.position[1])
			# self.goal_render = None  # Object(RED, self.object_size, self.goal[0], self.goal[1])
			# self.player_render = Object(BLUE, self.object_size, self.position[0], self.position[1])
			# self.goal_render = Object(RED, self.object_size, self.goal[0], self.goal[1])
		return self._update_state()

	def _reset_task(self):
		self.center = np.array([self.window_height / 2., self.window_width / 2.])
		self.position = np.array([self.window_height / 2., self.window_width / 2.])
		self.velocity = np.zeros(2)
		self.reward = 0
		self.value = 0

		# self.goal_degree = random.randint(0, 35) * (2*np.pi/36)
		self.goal_degree = random.random() * (2*np.pi)
		self.goal = np.array([self.rad*np.sin(self.goal_degree), self.rad*np.cos(self.goal_degree)])
		self.goal += self.center
		self.goal = np.round(self.goal) #New: goal position discretization

	def render(self, mode='console', close=False):
		if mode == 'console':
			print(self._update_state())
			pass

		elif mode == "human":
			# try:
			# 	# import pygame
			# 	from pygame import gfxdraw
			# 	import pygame.locals
			# except ImportError as e:
			# 	raise error.DependencyNotInstalled(
			# 		"{}. (HINT: install pygame using 'pip install pygame'".format(e)
			# 	)
			if close:
				pygame.quit()
			else:
				if self.screen is None:
					successes, failures = pygame.init()
					print("Initializing pygame: {0} successes and {1} failures.".format(successes, failures))
					self.screen = pygame.display.set_mode((self.window_width, self.window_height))
				if self.player_render is None:
					self.player_render = Object(BLUE, self.object_size, self.position[0], self.position[1])
				if self.goal_render is None:
					self.goal_render = Object(RED, self.object_size, self.goal[0], self.goal[1])

				self.screen.fill(BLACK)

				self.player_render.update(self.position[0], self.position[1])
				self.goal_render.update(self.goal[0], self.goal[1])
				self.screen.blit(self.player_render.image, self.player_render.rect)
				self.screen.blit(self.goal_render.image, self.goal_render.rect)
				pygame.display.update()

			if close:
				pygame.quit()


	def step(self, target):
		if self.done == 1:
			print("Game Over")
			return [self._update_state(), self.reward, self.done, None]
		else:
			self.count += 1
			info = {}
			# dt = 1./self.metadata["video.frames_per_second"]
			# dt = self.clock.tick(self.metadata["video.frames_per_second"]) / 1000
			# self.acceleration += target * dt
			# self.acceleration += self.calculate_fieldforce(self.velocity)
			# self.velocity += self.acceleration # temp
			target = np.clip(target, -1, 1)
			self.velocity = target * self.step_multiplier
			ff = self._calculate_fieldforce(self.velocity)
			self.velocity += ff
			self.position += self.velocity
			dist_goal = np.sqrt(((self.position - self.goal) ** 2).sum())

			value_cur = self.rew_multiplier * (1 - dist_goal/self.rad)
			self.reward = value_cur - self.value + self.rew_per_step # temp
			self.value = value_cur

			if dist_goal < self.dist_goal_thres:
				# print("near!")
				self.n_trial += 1
				self._reset_task()
				# self.done = 1		# done 신호가 이게 맞나?
			# print("position : ", self.position)
			# print("goal : ", self.goal)
			# print("reward : ", self.reward)
			# print("vel : ", self.velocity)
			# print("field force : ", ff)
			# print("goal : ", dist_goal, "reward : ", self.reward)

			self.render(self.mode)

			if self.count > self.count_max or self.n_trial >= self.n_max_trial:
				self.done = 1

		return [self._update_state(), self.reward, self.done, info]


	def _update_state(self):
		self.state = np.zeros((self.env_height, self.env_width, 3), dtype=np.uint8)
		posX, posY = np.round(self.position).astype(int)
		self.state[posY:posY + self.object_size, posX:posX + self.object_size] = BLUE
		posX_g, posY_g = np.round(self.goal).astype(int)
		self.state[posY_g:posY_g + self.object_size, posX_g:posX_g + self.object_size] = RED
		return self.state

	def _calculate_fieldforce(self, velocity):
		x, y = velocity
		phi = np.arctan2(y, x)
		F = np.array([np.sin(self.period_x * phi), -np.cos(self.period_y * phi)])
		mag_curr = self.magnitude * np.sqrt(x**2 + y**2)
		F = np.multiply(F, mag_curr)
		return F

	def set_mode(self, mode):
		self.mode = mode



class Object(pygame.sprite.Sprite):
	def __init__(self, c, size, posX, posY):
		super().__init__()
		self.image = pygame.Surface((size, size))
		self.image.fill(c)
		self.rect = self.image.get_rect()
		self.rect.center = (posX, posY)
		# self.velocity = [0, 0]

	def update(self, posX, posY):
		self.rect.center = (posX, posY)
		# self.rect.move(posX, posY)
		# self.rect.x = posX
		# self.rect.y = posY


successes, failures = pygame.init()
print("Initializing pygame: {0} successes and {1} failures.".format(successes, failures))
def main():
	env = FieldedMove()
	env.set_mode('human')
	# env.reset()
	running = True
	target = np.zeros(2)
	f = 1
	value = 0
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					target[1] -= f
				elif event.key == pygame.K_s:
					target[1] += f
				elif event.key == pygame.K_a:
					target[0] -= f
				elif event.key == pygame.K_d:
					target[0] += f
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_w:
					target[1] += f
				elif event.key == pygame.K_s:
					target[1] -= f
				elif event.key == pygame.K_a:
					target[0] += f
				elif event.key == pygame.K_d:
					target[0] -= f

		state, reward, done, info = env.step(target)
		value += reward
		if done:
			env.reset()
			print("cumulative reward : ", value)
			value = 0
		time.sleep(0.1)
		# plt.imshow(state)
		# plt.show()

	print("Exited the game loop. Game will quit...")
	quit()

if __name__ == "__main__":
	main()
