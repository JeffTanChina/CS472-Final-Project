#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import math
import sys
import re
from math import log
from math import exp
from math import sqrt
import time

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [float(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0


    for _ in range(MAX_ITERS):
        dw = [0.0] * numvars
        db = 0.0
        avg_loss = 0
        for (x, y) in data:
            z = predict_lr((w, b), x)
            loss = logisticloss(z, y)
            # Update gradients
            db += loss * (-y)

            for i in range(numvars):
                dw[i] += -y*loss * x[i]

            # cross-entropy
            avg_loss+=loss
        print(avg_loss/len(data))
        # Apply regularization to gradients
        for i in range(numvars):
            dw[i] += l2_reg_weight * w[i]

        # Update model parameters
        b = b - eta * db
        for i in range(numvars):
            w[i] -= eta * dw[i]
        #print(w)
        # converge
        magnitude = 0
        for i in range(numvars):
            magnitude += dw[i] * dw[i]
        magnitude = math.sqrt(magnitude)

        if magnitude < 0.0001:
            break

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    #
    # YOUR CODE HERE
    #
    z = b
    for i in range(len(w)):
        z += x[i] * w[i]
    return z


def logistic(z):
    try:
        pred = 1.0 / (1.0 + exp(-z))
    except OverflowError:
        pred = 0.000000001
    return pred


def logisticloss(z, y):
    try:
        pred = 1.0 / (1.0 + exp(y*z))
    except OverflowError:
        pred = 0.00000001
    return pred


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    start_time = time.time()  # Measure train time
    (w, b) = train_lr(train, eta, lam)
    train_time = time.time() - start_time  # Measure train time
    print("Train time:", train_time)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    start_time = time.time()  # Measure test time
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        # print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)
    test_time = time.time() - start_time  # Measure test time
    print("Test time:", test_time)


if __name__ == "__main__":
    main(sys.argv[1:])
