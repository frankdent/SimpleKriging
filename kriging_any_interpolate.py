'''
Simple Kriging using any number of known and a grid of unknown points

Assumption: Let us assume that the empirical semivariogram is represented
by the linear regression line with Y intercept = 0 and slope = 4.0
'''

__author__ = "François d'Entremont"

import numpy as np
import matplotlib.pyplot as plt
import random

# Define the data in the form of a list of tuples
# k for known points xyz and u for unknown points xy
k = [(random.uniform(0, 20), random.uniform(0, 20), random.uniform(100, 200)) for _ in range(50)]
u=[]
for j in range(200):
    u += [(0.1*i, 0.1*j) for i in range(200)]
x_u_fill = [point[0] for point in u]
y_u_fill = [point[1] for point in u]

def distance(p1, p2):
    '''
    This function calculates the distance between two points.
    Keyword arguments:
    p1, p2 (tuple, list): coordinate with x and y as first and second indices. 
    '''
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**(1/2)

distance_matrix = np.zeros((len(k), len(k))) # Creating empty matrix
for i in range(len(k)): # Looping through the k rows of the matrix
    for j in range(len(k)): # Looping through the elements in each row
        distance_matrix[i, j] = distance(k[i], k[j])

semivariance_matrix = 4*distance_matrix

distance_to_unknowns = np.zeros((len(u), len(k))) # Creating empty matrix
for i in range(len(u)):
    for j in range(len(k)):
        distance_to_unknowns[i, j] = distance(u[i], k[j])

semivariance_to_unknowns = 4*distance_to_unknowns

# Assembling Gamma and appending Lagrange multipliers
# by adding ones to the right side and bottom and then
# setting the diagonal to zeros
gamma = np.append(semivariance_matrix, np.ones(
    [semivariance_matrix.shape[0], 1]), axis=1)
gamma = np.append(gamma, np.ones([1, gamma.shape[1]]), axis=0)
np.fill_diagonal(gamma, 0)

# Assembling vector Beta and appending Lagrange multipliers
beta = np.append(semivariance_to_unknowns, 
                 np.ones([len(u), 1]), axis=1).transpose()

# Calculating lambda: 
lambda_vector = np.matmul(np.linalg.inv(gamma), beta)

# Finding the variance
variance = np.zeros([len(u), 1])
for i in range(len(u)):
    for j in range(len(k)):
        variance[i][0] += lambda_vector[j][i]*semivariance_to_unknowns[i][j]
# Finding the standard error
std_error = np.sqrt(variance)

# Print known points to the console
for i in range(len(k)):
    print(f'k{i} = {k[i]}')

# Assembling results vector containing elevations for unknown points
# and printing the results to the console
results = np.zeros([len(u), 1])
for i in range(len(u)):
    for j in range(len(k)):
        results[i][0] += lambda_vector[j][i]*k[j][2] 
    print(f'u{i} = ({u[i][0]}, {u[i][1]}, {results[i][0]}), ' +
          f'variance = {round(variance[i][0], 2)}, standard error = '+
          f'{round(std_error[i][0], 2)}, 95% CI = ' +
          f'({round(results[i][0] - 1.96*std_error[i][0], 2)}, ' +
          f'{round(results[i][0] + 1.96*std_error[i][0], 2)})')

# Plotting the results
fig, ax = plt.subplots()
scat_fill = ax.scatter(x_u_fill, y_u_fill, c=results, cmap='jet')
scat_fill.set_clim(100, 200)
x_known = [point[0] for point in k]
y_known = [point[1] for point in k]
plt.scatter(x_known, y_known, s=20)
plt.title('Scatterplot of x vs y')
plt.xlabel("x")
plt.ylabel("y", rotation='horizontal')
for i in range(len(k)): #adding elevation labels for known points
    plt.annotate(round(k[i][2], 2), (round(k[i][0], 2),round(k[i][1], 2))) 
plt.show()