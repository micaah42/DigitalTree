import random
import numpy as np

from utils import logistic
from tree import Tree, TreeBrain, TreeNode


class NoBrain(TreeBrain):
    def __init__(self):
        self.happy = True

    def describe(self):
        return 'NoBrain(olEeoleoleoleeee)'

    def mutate(self, lr: float = 0.01) -> 'TreeBrain':
        return NoBrain()

    def wants_to_grow(self, here: TreeNode) -> float:
        return random.random()

    def angle(self, here: TreeNode) -> int:
        angle = np.random.randint(0, TreeNode.angles)
        return angle

    def ratio(self, here: TreeNode) -> float:
        return random.random()


class SimpleBrain(TreeBrain):
    def __init__(self, weights: np.ndarray = None):
        self.happy = True
        self.weights = np.zeros((TreeNode.features_shape + 2, 3)) if weights is None else weights

    def feed_features(self, x: np.ndarray, row: int):
        return logistic(self.weights[-1, row] * np.dot(x, self.weights[:-2, row]) + self.weights[-2, row])

    def describe(self):
        return f'SimpleBrain({self.weights.shape})'

    def wants_to_grow(self, here: TreeNode) -> float:
        return self.feed_features(here.features(), 0)

    def angle(self, here: TreeNode) -> int:
        return int((TreeNode.angles - 1) * self.feed_features(here.features(), 1))

    def ratio(self, here: TreeNode) -> float:
        return self.feed_features(here.features(), 2)

    def mutate(self, lr: float = 0.01):
        weights = self.weights + (np.random.random(self.weights.shape) - 0.5) * lr
        return SimpleBrain(weights)