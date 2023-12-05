# Define the desired distribution to sample from
# GUSSIAN SAMPLING...
# Import Libraries

d = 2 # Number of dimensions
mean = np.matrix([[0.], [1.]])
covariance = np.matrix([
[1, 0.8],
[0.8, 1]
])

# Compute the Decomposition:
A = np.linalg.cholesky(covariance)
 
# Sample X from standard normal
n = 50 # Samples to draw
Z = np.random.normal(size=(d, n))
 
# Apply the transformation
X = A.dot(Z) + mean
 
# Plot the samples and the distribution
fig, ax = plt.subplots(figsize=(6, 4.5))
# Plot bivariate distribution
x1, x2, p = generate_surface(mean, covariance, d)
con = ax.contourf(x1, x2, p, 33, cmap=cm.YlGnBu)
# Plot samples
ax.plot(Y[0,:], Y[1,:], 'ro', alpha=.6,
        markeredgecolor='k', markeredgewidth=0.5)
ax.set_xlabel('y1', fontsize=13)
ax.set_ylabel('y2', fontsize=13)
ax.axis([-2.5, 2.5, -1.5, 3.5])
ax.set_aspect('equal')
ax.set_title('Samples from bivariate normal distribution')
cbar = plt.colorbar(con)
cbar.ax.set_ylabel('density: p(y1, y2)', fontsize=13)
plt.show()
