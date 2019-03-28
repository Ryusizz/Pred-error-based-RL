import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class FieldedMove(gym.Env):
	metadata = {'render.modes': ['human']}


	def __init__(self):
		# self.state = []
		# for i in range(3):
		# 	self.state += [[]]
		# 	for j in range(3):
		# 		self.state[i] += ["-"]
		# self.counter = 0
		# self.done = 0
		# self.add = [0, 0]
		# self.reward = 0
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
		self.action = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
		self.position = [0., 0.]
		self.velocity = [0., 0.]
		self.acceleration = [0., 0.]
		self.counter = 0
		self.done = 0
		self.reward = 0

		self.goal = [1., 1.] # temp
		self.magnitude = [15, 15]
		self.field = [1., 1.]


	# def check(self):
	#
	# 	if(self.counter<5):
	# 		return 0
	# 	for i in range(3):
	# 		if(self.state[i][0] != "-" and self.state[i][1] == self.state[i][0] and self.state[i][1] == self.state[i][2]):
	# 			if(self.state[i][0] == "o"):
	# 				return 1
	# 			else:
	# 				return 2
	# 		if(self.state[0][i] != "-" and self.state[1][i] == self.state[0][i] and self.state[1][i] == self.state[2][i]):
	# 			if(self.state[0][i] == "o"):
	# 				return 1
	# 			else:
	# 				return 2
	# 	if(self.state[0][0] != "-" and self.state[1][1] == self.state[0][0] and self.state[1][1] == self.state[2][2]):
	# 		if(self.state[0][0] == "o"):
	# 			return 1
	# 		else:
	# 			return 2
	# 	if(self.state[0][2] != "-" and self.state[0][2] == self.state[1][1] and self.state[1][1] == self.state[2][0]):
	# 		if(self.state[1][1] == "o"):
	# 			return 1
	# 		else:
	# 			return 2

	def getstate(self):
		return self.position

	def calculate_fieldforce(self, velocity):
		xd, yd = velocity
		phi = arctan(yd/xd)
		F = [-sin(self.period * phi), cos(self.period * phi)]
		F = -15 * sqrt(xd^2 + yd^2)
		return F


	def step(self, target):
		if self.done == 1:
			print("Game Over")
			return [self.getstate(), self.reward, self.done, None]
		else:
			self.acceleration += target
			self.acceleration += self.calculate_fieldforce(self.velocity)
			self.velocity += self.acceleration # temp

			self.render()

		win = self.check()
		if(win):
			self.done = 1;
			print("Player ", win, " wins.", sep = "", end = "\n")
			self.add[win-1] = 1;
			if win == 1:
				self.reward = 100
			else:
				self.reward = -100

		return [self.state, self.reward, self.done, self.add]

	def reset(self):
		for i in range(3):
			for j in range(3):
				self.state[i][j] = "-"
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0
		return self.state

	def render(self):
		for i in range(3):
			for j in range(3):
				print(self.state[i][j], end = " ")
			print("")
