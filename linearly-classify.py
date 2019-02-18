#
# CSE 5522: HW 3 Linear Classification w/ Perceptrons
# Author: Nora Myer
#

from numpy import *
from matplotlib.pyplot import *
import json

data_vectorization_labels = ["is_weekday", "is_Saturday", "is_Sunday", "is_morning", "is_afternoon", "is_evening", "is_<30", "is_30-60", "is_>60", "is_silly", "is_happy", "is_tired", "friendsVisiting", "kidsPlaying", "atHome", "snacks"]
training_epochs = 10

def plot_assignments(curmeans,data,labels):
    clf()
    curassign = kmeans_pointassignments(curmeans,data)

    for i in range(0 , curmeans.shape[0]):
        tp = compress(curassign == i, data, axis = 0)
        plot(tp[:,0],tp[:,1],colors[i])
    for ((x,y),lab) in zip(data ,labels):
        text(x + .03, y + .03, lab, fontsize = 9)
    plot(curmeans[:,0], curmeans[:,1], 'c^', markersize = 12)
    show(block = block_with_plot)

def read_data(training_file, test_file):
    training_set = np.genfromtxt(training_file, delimiter=',', dtype='str')
    test_set = np.genfromtxt(test_file, delimiter=',', dtype='str')
    return training_set, test_set

def vectorize_data(data_set):
    vectorized_data = []
    for row in data_set:
        vectorized_row = []
        for idx, attr in enumerate(row):
            if idx is 0:
                if row[idx] == "Weekday":
                    vectorized_row.append(1)
                    vectorized_row.append(0)
                    vectorized_row.append(0)


    return vectorized_data, vectorized_labels

def averaged_perceptron(train_set, train_labels, test_data, test_labels):
    w = np.zeros(len(train_set[0]))
    t = np.zeros(len(train_set[0]))

    for l in range(training_epochs):
        for i in range(len(train_set)):
            h = 0
            if np.dot(w, train_set[i]) >= 0:
                h = 1

            w = w + ((train_labels[i] - h) * train_set[i])
            t = t + w

    return 1.0/(training_epochs * len(train_set)) * t

#computes the accuracy of classify_test_documents with correct results
def get_accuracy(predictions, accurate_classifications):
    inaccurate_count = 0

    for i in range(len(predictions)):
        if predictions[i] != accurate_classifications[i]:
            inaccurate_count += 1

    return 1.0 - (float(inaccurate_count) / len(predictions))

def main():
    training_set, test_set = read_data("game_attrdata_train.dat", "game_attrdata_test.dat")

if __name__ == "__main__":
    main()
