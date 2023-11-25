# Import Libreries...

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

def get_gaussian_random():
    m = 0
    while m == 0:
        m = round(np.random.random() * 100)

    numbers = np.random.random(int(m))
    summation = float(np.sum(numbers))
    gaussian = (summation - m / 2) / math.sqrt(m / 12.0)

    return gaussian

def learn_mean_cov(pts):
    learned_mean = np.matrix([[0.0], [0.0]])
    learned_cov = np.zeros((2, 2))
    count = len(pts)
    for pt in pts:
        learned_mean += pt
        learned_cov += pt * pt.transpose()

    learned_mean /= count
    learned_cov /= count
    learned_cov -= learned_mean * learned_mean.transpose()
    return learned_mean, learned_cov


def generate_known_gaussian(dimensions, count):
    ret = []
    for i in range(count):
        current_vector = []
        for j in range(dimensions):
            g = get_gaussian_random()
            current_vector.append(g)

        ret.append(tuple(current_vector))

    return ret

def sample(count, c):
    known = generate_known_gaussian(2, count)
    target_mean = np.matrix([[2.0], [8.0]])
    target_cov = np.matrix([[0.25, 0.25],
                            [0.25, 0.5]])

    [eigenvalues, eigenvectors] = np.linalg.eig(target_cov)
    l = np.matrix(np.diag(np.sqrt(eigenvalues)))
    Q = np.matrix(eigenvectors) * l
    x1_tweaked = []
    x2_tweaked = []
    tweaked_all = []
    for i, j in known:
        original = np.matrix([[i], [j]]).copy()
        tweaked = (Q * original) + target_mean
        x1_tweaked.append(float(tweaked[0]))
        x2_tweaked.append(float(tweaked[1]))
        tweaked_all.append(tweaked)
    if c == 0:
        mu = np.array([2.0, 3.0])
        X, Y = np.meshgrid(x1_tweaked, x2_tweaked)
        pos = np.dstack((X, Y))
        rv = multivariate_normal(mu, target_cov)
        Z = rv.pdf(pos)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
    #    fig.show()
    return tweaked_all
    
known = generate_known_gaussian(2,100)
data = np.asarray(known)
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)
print(x.shape)
mean1 = (2, 3)
cov1 = ([0, 0.5],
        [0.5, 0.25])
Ax1 = np.random.multivariate_normal(mean1, cov1, 100)


temps = data[:, 0]
rentals = data[:, 1]

plt.scatter(temps, rentals, marker='x', color='red')
plt.xlabel('Normalized Temperature in C')
plt.ylabel('Bike Rentals in 1000s')

def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))


theta = np.zeros(2)
X = np.column_stack((np.ones(len(temps)), temps))
y = rentals
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)

def gradient_descent(X, y, alpha, iterations):
    theta = np.zeros(2)
    m = len(y)

    for i in range(iterations):
        t0 = theta[0] - (alpha / m) * np.sum(np.dot(X, theta) - y)
        t1 = theta[1] - (alpha / m) * np.sum((np.dot(X, theta) - y) * X[:, 1])
        theta = np.array([t0, t1])

    return theta


iterations = 5000
alpha = 0.1

theta = gradient_descent(X, y, alpha, iterations)
cost = compute_cost(X, y, theta)

print("theta:", theta)
print('cost:', compute_cost(X, y, theta))

plt.scatter(temps, rentals, marker='x', color='red')
plt.xlabel('Normalized Temperature in C')
plt.ylabel('Bike Rentals in 1000s')
samples = np.linspace(min(temps), max(temps))
plt.plot(samples, theta[0] + theta[1] * samples)

Xs, Ys = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-40, 40, 50))
Zs = np.array([compute_cost(X, y, [t0, t1]) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = np.reshape(Zs, Xs.shape)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection="3d")
ax.set_xlabel(r't0')
ax.set_ylabel(r't1')
ax.set_zlabel(r'cost')
ax.view_init(elev=25, azim=40)
ax.plot_surface(Xs, Ys, Zs, cmap=cm.rainbow)

ax = plt.figure().gca()
ax.plot(theta[0], theta[1], 'r*', label='Solution Found')
CS = plt.contour(Xs, Ys, Zs, np.logspace(-10, 10, 50), label='Cost Function')
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Contour Plot of Cost Function")
plt.xlabel("w0")
plt.ylabel("w1")
plt.legend(loc='best')
plt.show()
