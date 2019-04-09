import os
import sys
import random

# import curses
import pygame
import pygame.locals
from pygame import gfxdraw
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

FPS = 10
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class FieldedMove(gym.Env):
	metadata = {'render.modes': ['console', 'human'],
				'video.frames_per_second' : 60}

	def __init__(self, mode):
		self.window_height = 84
		self.window_width = 84
		self.env_height = 84
		self.env_width = 84
		self.object_size = 7
		self.rad = 25

		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.env_height, self.env_width, 3), dtype=np.uint8)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
		self.center = None #np.array([self.window_height/2., self.window_width/2.])
		self.position = None #np.array([self.window_height/2., self.window_width/2.])
		self.velocity = None #np.zeros(2)
		# self.acceleration = np.zeros(2)

		self.count = None
		self.count_max = 100
		self.done = 0
		self.value = None
		self.reward = None
		self.rew_per_step = -0.01

		# self.goal = np.ones(2) # temp
		self.dist_goal_thres = 1.5
		self.goal_degree = None #2*np.pi
		self.goal = None #np.array([self.rad * np.sin(self.goal_degree), self.rad * np.cos(self.goal_degree)])
		# self.goal += self.center

		self.magnitude = 0.3 #np.array([15, 15])
		self.field = np.array([1., 1.])
		self.period = None #random.randint(1,3)

		self.mode = mode
		if self.mode == 'human':
			successes, failures = pygame.init()
			print("Initializing pygame: {0} successes and {1} failures.".format(successes, failures))

			self.screen = pygame.display.set_mode((self.window_height, self.window_width))
			self.clock = pygame.time.Clock()

			self.player_render = None #Object(BLUE, self.object_size, self.position[0], self.position[1])
			self.goal_render = None #Object(RED, self.object_size, self.goal[0], self.goal[1])
			# self.running = True


	def _update_state(self):
		self.state = np.zeros((self.env_height, self.env_width, 3), dtype=np.uint8)
		posX, posY = np.round(self.position).astype(int)
		self.state[posY:posY+self.object_size, posX:posX+self.object_size] = BLUE
		posX_g, posY_g = np.round(self.goal).astype(int)
		self.state[posY_g:posY_g+self.object_size, posX_g:posX_g+self.object_size] = RED
		return self.state

	def calculate_fieldforce(self, velocity):
		x, y = velocity
		phi = np.arctan(y/(x+1e-15))
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
			dt = clock.tick(FPS) / 1000
			# self.acceleration += target * dt
			# self.acceleration += self.calculate_fieldforce(self.velocity)
			# self.velocity += self.acceleration # temp
			target = np.clip(target, -20, 20)
			self.velocity = target * dt
			ff = self.calculate_fieldforce(self.velocity)
			self.velocity += ff
			self.position += self.velocity
			dist_goal = np.sqrt(((self.position - self.goal) ** 2).sum())

			value_cur = 1 - dist_goal/self.rad
			self.reward = value_cur - self.value + self.rew_per_step # temp
			self.value = value_cur

			if dist_goal < self.dist_goal_thres:
				self.done = 1		# done 신호가 이게 맞나?
			print("position : ", self.position)
			print("goal : ", self.goal)
			print("reward : ", self.reward)
			print("vel : ", self.velocity)
			print("field force : ", ff)
			print("goal : ", dist_goal, "reward : ", self.reward)

			self.render(self.mode)


		return [self._update_state(), self.reward, self.done, None]

	def reset(self):
		self.count = 0
		self.done = 0
		self.center = np.array([self.window_height / 2., self.window_width / 2.])
		self.position = np.array([self.window_height / 2., self.window_width / 2.])
		self.velocity = np.zeros(2)
		self.reward = 0
		self.value = 0

		self.period = random.randint(1, 4)

		self.goal_degree = random.randint(0, 35) * (2*np.pi/36)
		self.goal = np.array([self.rad*np.sin(self.goal_degree), self.rad*np.cos(self.goal_degree)])
		self.goal += self.center
		self.reward = 0
		if self.mode == "human":
			self.player_render = Object(BLUE, self.object_size, self.position[0], self.position[1])
			self.goal_render = Object(RED, self.object_size, self.goal[0], self.goal[1])
		return self._update_state()

	def render(self, mode='console', close=False):
		if mode == 'console':
			print(self._update_state())

		elif mode == "human":
			try:
				screen.fill(BLACK)

				self.player_render.update(self.position[0], self.position[1])
				screen.blit(self.player_render.image, self.player_render.rect)
				screen.blit(self.goal_render.image, self.goal_render.rect)
				pygame.display.update()

			except ImportError as e:
				raise error.DependencyNotInstalled(
					"{}. (HINT: install pygame using `pip install pygame`".format(e))
			if close:
				pygame.quit()

successes, failures = pygame.init()
print("Initializing pygame: {0} successes and {1} failures.".format(successes, failures))

screen = pygame.display.set_mode((720, 480))
clock = pygame.time.Clock()


class Object(pygame.sprite.Sprite):
	def __init__(self, c, size, posX, posY):
		super().__init__()
		self.image = pygame.Surface((size, size))
		self.image.fill(c)
		self.rect = self.image.get_rect()
		self.rect.topleft = (posX, posY)
		# self.velocity = [0, 0]

	def update(self, posX, posY):
		self.rect.topleft = (posX, posY)
		# self.rect.move(posX, posY)
		# self.rect.x = posX
		# self.rect.y = posY


def main():
	env = FieldedMove('human')
	env.reset()
	running = True
	target = np.zeros(2)
	f = 15
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
		if done:
			env.reset()
		# plt.imshow(state)
		# plt.show()

	print("Exited the game loop. Game will quit...")
	quit()

if __name__ == "__main__":
	main()
