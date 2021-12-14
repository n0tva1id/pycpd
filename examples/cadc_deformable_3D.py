from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
from pycpd import RigidRegistration, AffineRegistration
import numpy as np


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
        # X = np.loadtxt('data/bunny_target.txt')
        # # synthetic data, equaivalent to X + 1
        # Y = np.loadtxt('data/bunny_source.txt')
    pcl_a = np.fromfile(
        "/Users/exthardwaremac/Desktop/cadc_seq/0000000000.bin", dtype=np.float32).reshape((-1, 4))

    pcl_b = np.fromfile(
        "/Users/exthardwaremac/Desktop/cadc_seq/0000000004.bin", dtype=np.float32).reshape((-1, 4))

    max_length = min(pcl_a.shape[0], pcl_b.shape[0])

    pcl_a = pcl_a[:max_length]
    pcl_b = pcl_b[:max_length]

    pcl_a = pcl_a[:2000]
    pcl_b = pcl_b[:2000]

    pcl_a.tofile("before_align_a_first2000.bin")
    pcl_b.tofile("before_align_b_first2000.bin")
    
    pcl_a = pcl_a[:, :3]
    pcl_b = pcl_b[:, :3]

    print(pcl_a.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    # reg = DeformableRegistration(**{'X': pcl_a, 'Y': pcl_b, 'low_rank': True})
    # reg = RigidRegistration(**{'X': pcl_a, 'Y': pcl_b})
    reg = AffineRegistration(**{'X': pcl_a, 'Y': pcl_b})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
