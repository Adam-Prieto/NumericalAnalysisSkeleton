import math
import random
import time

import numpy
import numpy as np
import scipy.linalg

e = math.e
pi = math.pi

# Given matrices -- A-D are 3x3 while E is 2x2
A = numpy.array([[1, 2, 3], [-2, 3, 1], [0, 4, 1]])
B = numpy.array([[-2, 0, 1], [1, 3, 0], [3, 4, 2]])
C = numpy.array([[1, -1, 2], [-1, 0, 3], [2, 3, 1]])
D = numpy.array([[2, 1, 2], [1, 0, 1], [4, 1, 4]])
E = numpy.array([[1, 4], [6, 3]])  # 2x2
matrices = [A, B, C, D, E]

# New stuff...
F = numpy.array([[3, 1, 2], [6, 3, 4], [3, 1, 5]])
I, L, U = scipy.linalg.lu(F)


# # Print original matrix
# print(f"Matrix: ")
# pprint.pprint(F)
#
# # Print lower matrix
# print("L:")
# pprint.pprint(L)
#
# # Print upper matrix
# print("U:")
# pprint.pprint(U)


# ****************************************************************************
# Specified HW4 functions:

def f1(x):
    return x ** 3 - 2 * x - 2


def f2(x):
    return e ** x + x - 7


def f3(x):
    return e ** x + math.sin(x) - 4


functions = [f1, f2, f3]


# ****************************************************************************

# @methodName - secantMethod
# @param: x0 - first initial guess on where a root lies.
# @param: x1 - second initial guess on where a root lies.
# @param: function - a function to find a numeric root.
# @param: steps - the number of steps/ iterations to run the loop.
# @return - a numeric value based on where a numeric root lies.

def secantMethod(x0, x1, function, steps):
    # Place holder variable
    result = 0

    # Run loop as long as there are steps remaining
    while steps > 0:

        # Error checking
        if function(x0) == function(x1):
            print(f"Divide by zero error with {steps} step(s) left.")
            break

        # Do secant method calculation
        x2 = x0 * function(x1) - x1 * function(x0)
        x2 /= (function(x1) - function(x0))
        result = x2
        steps -= 1
        x0 = x1
        x1 = x2

    return result


# ****************************************************************************
# Adam Prieto
# Doctor Robert Niemeyer
# Numerical Analysis I
# 9 December 2022
# Week 15 Homework Computer Problem(s)
# Video Link:
# Description: Use Cholesky factorization to factor the matrix from the first
#              problem on the written portion of the homework.


# a = numpy.array([[2, 2], [2, 5]])
# b = numpy.array([[16, 4], [0, 2]])
# cholesky1 = numpy.linalg.cholesky(a)
# cholesky2 = numpy.linalg.cholesky(b)
# print(cholesky1)
# print(cholesky2)

# ****************************************************************************

# Adam Prieto
# Doctor Robert Niemeyer
# Numerical Analysis II
# 10 February 2023
# Homework 1 Computer Problem(s)
# Description: Write a program in Python that calculates the interpolating
# polynomial using the method of Lagrange and the method of Newton.


def lagrangeInterpolation(xValues, yValues, xEval):
    n = len(xValues)
    y_eval = 0

    # Calculate the ith Lagrange basis polynomial
    for i in range(n):
        l = 1
        for j in range(n):
            if j != i:
                l *= (xEval - xValues[j]) / (xValues[i] - xValues[j])
        y_eval += yValues[i] * l
    return y_eval


def newtonInterpolation(x, y):
    n = len(x)
    m = np.zeros((n, n))
    m[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            m[i, j] = (m[i, j - 1] - m[i - 1, j - 1]) / (x[i] - x[i - j])
    c = np.zeros(n)
    for i in range(n):
        c[i] = m[i, i]

    def p(x_eval):
        """Evaluates the Newton polynomial at x_eval."""
        y_eval = c[n - 1]
        for i in range(n - 2, -1, -1):
            y_eval = y_eval * (x_eval - x[i]) + c[i]
        return y_eval

    return p


# Define & print data points
x = []
y = []
for i in range(1, 100):
    x.append(i)
    y.append((i - i / random.randint(1, 6)))

for i in range(len(x)):
    print(f"({x[i]}, {y[i]})")
print("\n")

# Lagrange Evaluation
start = time.time()
lagrangeResult = lagrangeInterpolation(x, y, 0)
finish = time.time()
print(f"Lagrange Interpolation:\nf(0) = {round(lagrangeResult)}\nTime elapsed: "
      f"{finish - start} sec.\n\n")

# Newton Evaluation
start = time.time()
p = newtonInterpolation(x, y)
newtonResult = p(0)
finish = time.time()
print(f"Newton Interpolation:\nf(0) = {round(newtonResult)}\nTime elapsed: "
      f"{finish - start} sec.")
