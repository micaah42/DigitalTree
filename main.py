import random
import numpy as np
import matplotlib.pyplot as plt

from tree import *
from brains import *


def main():
    trees = []
    num_trees = 1000
    for i in range(num_trees):
        # grow tree
        tree = Tree(NoBrain())
        for _ in range(5):
            tree.grow()
        trees.append(tree)
        print(f'\rgrowing no brain trees...   | {100 * (i + 1) / num_trees:.1f}%', end='')

    best = trees[0]
    print('\nevaluating no brain trees...')
    for tree in trees:
        if tree.gain() > best.gain():
            best = tree
    print(f'Best NoBrain: {best.gain()}')
    best.draw()
    plt.title('NoBrain')
    plt.show()

    epochs = 1000
    plot_epochs = [100, 200, 500, 750]
    num_children = 10
    num_survivors = 5
    epoch_trees = [Tree(SimpleBrain())]
    for epoch in range(epochs):
        survivors = epoch_trees[:num_survivors]
        epoch_trees = []
        for survivor in survivors:
            for _ in range(num_children):
                tree = Tree(survivor.brain.mutate())
                for _ in range(5):
                    tree.grow()
                epoch_trees.append(tree)

        epoch_trees = sorted(epoch_trees, key=lambda tree: -tree.gain())
        if epoch in plot_epochs:
            epoch_trees[0].draw()
            plt.title(f'{epoch_trees[0].brain.describe()} - {epoch}')
            plt.show()
        print(f'\rgain: {epoch_trees[0].gain():.1f} {100 * (epoch + 1) / epochs:.0f}%', end='')

    best_tree = epoch_trees[0]
    print(f'\nBest SimpleBrain: {best_tree.gain()}')
    print(best_tree.brain.describe())
    best_tree.draw()
    plt.title('SimpleBrain')
    plt.show()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
