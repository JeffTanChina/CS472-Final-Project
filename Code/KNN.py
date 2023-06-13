import math
import sys
import re
from collections import Counter
import time

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


# Train a K-Nearest Neighbors model
def train_knn(data, k):
    return data


# Predict the label for a given example using the trained KNN model
def predict_knn(model, example):
    k = len(model)
    distances = []
    for (x, y) in model:
        dist = euclidean_distance(x, example)
        distances.append((dist, y))
    distances.sort(key=lambda x: x[0])  # Sort by distance
    neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in neighbors]
    counts = Counter(labels)
    majority_label = counts.most_common(1)[0][0]
    return majority_label


# Calculate the Euclidean distance between two vectors
def euclidean_distance(v1, v2):
    squared_diffs = [(x1 - x2) ** 2 for (x1, x2) in zip(v1, v2)]
    return math.sqrt(sum(squared_diffs))


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: knn.py <train> <test> <k>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    k = int(argv[2])

    # Train model
    start_time = time.time()  # Measure train time
    model = train_knn(train, k)
    train_time = time.time() - start_time  # Measure train time
    print("Train time:", train_time)

    # Write model file (not necessary for KNN)

    # Make predictions, compute accuracy
    start_time = time.time()  # Measure test time
    correct = 0
    for (x, y) in test:
        pred = predict_knn(model, x)
        if pred == y:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy:", acc)
    test_time = time.time() - start_time  # Measure test time
    print("Test time:", test_time)


if __name__ == "__main__":
    main(sys.argv[1:])
