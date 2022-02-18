import math
import typing as tp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class TreeNode:
    angles = 6
    features_shape = 5

    def __init__(self, x: float, y: float, angle: int, ratio: float):
        radians = 2 * angle * math.pi / TreeNode.angles
        self.x = ratio * math.cos(radians) + x
        self.y = ratio * math.sin(radians) + y

        self.ratio = ratio
        self._children: tp.List[tp.Optional[TreeNode]] = [None for _ in range(self.angles)]

    def grow(self, angle: int, ratio: float):
        assert 0 <= angle < TreeNode.angles, f'{angle}'
        assert 0 <= ratio <= 1
        if self._children[angle] is None:
            self._children[angle] = TreeNode(self.x, self.y, angle, ratio)

    def gain(self) -> float:
        s = self.y * (1. - self.ratio)
        for child in self.children():
            s += child.gain()
        return s

    def ancestors(self) -> tp.List['TreeNode']:
        c = [self]
        for child in self.children():
            c.extend(child.ancestors())
        return c

    def children(self) -> tp.List['TreeNode']:
        c = []
        for child in self._children:
            if child is not None:
                c.append(child)
        return c

    def features(self) -> np.ndarray:
        return np.array([self.x, self.y, len(self.children()), len(self.ancestors()), 1.])

    def draw(self):
        for child in self.children():
            plt.plot([self.x, child.x], [self.y, child.y])
            child.draw()


class TreeBrain(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def describe(self):
        pass

    @abstractmethod
    def mutate(self, lr: float = 0.01) -> 'TreeBrain':
        pass

    @abstractmethod
    def wants_to_grow(self, here: TreeNode) -> float:
        pass

    @abstractmethod
    def angle(self, here: TreeNode) -> int:
        pass

    @abstractmethod
    def ratio(self, here: TreeNode) -> float:
        pass


class Tree:
    def __init__(self, brain: TreeBrain):
        self.root = TreeNode(0, 0, TreeNode.angles // 2, 1)
        self.brain = brain

    def grow(self):
        # determine how many nodes we can grow
        nodes_to_grow = self.gain() / (1. + self.gain()) * 2 + 1

        nodes = self.root.ancestors()
        growth_needs = list(map(lambda x: -self.brain.wants_to_grow(x), nodes))
        limit = sorted(growth_needs)[max(0, min(len(nodes) - 1, int(nodes_to_grow)))]
        for i, node in enumerate(nodes):
            if self.brain.wants_to_grow(node) > limit:
                nodes[i].grow(self.brain.angle(node), self.brain.ratio(node))

    def gain(self) -> float:
        return self.root.gain()

    def draw(self):
        self.root.draw()
