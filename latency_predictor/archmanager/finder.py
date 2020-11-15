import copy
import numpy.random as random
from tqdm import tqdm
import numpy as np




class myArchManager:
	def __init__(self):
		self.num_blocks = 40
		self.num_stages = 10
		self.kernel_sizes = [3, 5, 7]
		self.expand_ratios = [3, 4, 6]
		self.depths = [1, 2, 3, 4]
		self.resolutions = [160, 320, 512, 768, 1024]

	def random_sample(self):
		sample = {}
		d = []
		e = []
		ks = []
		for i in range(self.num_stages):
			d.append(random.choice(self.depths,p=[0.25]*4))

		for i in range(self.num_blocks):
			e.append(random.choice(self.expand_ratios))
			ks.append(random.choice(self.kernel_sizes))

		sample = {
			'wid': None,
			'ks': ks,
			'e': e,
			'd': d,
			'r': [random.choice(self.resolutions,p=[0.2, 0.2, 0.2, 0.2,0.2])]
		}


		return sample