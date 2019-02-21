#
# CSE 5522: HW 3 Linear Classification w/ Perceptrons
# Author: Nora Myer
#

from numpy import *
from matplotlib.pyplot import *
import json

data_vectorization_labels = [1.0, "is_weekday", "is_Saturday", "is_Sunday", "is_morning", "is_afternoon", "is_evening", "is_<30", "is_30-60", "is_>60", "is_silly", "is_happy", "is_tired", "friendsVisiting", "kidsPlaying", "atHome", "snacks"]
training_epochs = 10
attr_dict = {}

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

def build_attr_dict():
    global attr_dict
    attr_dict[0] = {'Weekday': 0.0, 'Saturday': 0.0, 'Sunday': 0.0}
    attr_dict[1] = {'morning': 0.0, 'afternoon': 0.0, 'evening': 0.0}
    attr_dict[2] = {'<30': 0.0, '30-60': 0.0, '>60': 0.0}
    attr_dict[3] = {'silly': 0.0, 'happy': 0.0, 'tired': 0.0}
    attr_dict[4] = {'no': 0.0, 'yes': 0.0}
    attr_dict[5] = {'no': 0.0, 'yes': 0.0}
    attr_dict[6] = {'no': 0.0, 'yes': 0.0}
    attr_dict[7] = {'no': 0.0, 'yes': 0.0}

def vectorize_data(data_set):
    label_idx = len(data_set[0]) - 1
    vectorized_data = []
    vectorized_labels = [1 if x[label_idx] == "SettersOfCatan" else 0 for x in data_set]

    for row in data_set:
        vectorized_row = [1.0]
        for idx, attr in enumerate(row):
            if not idx is label_idx:
                attr_dict[idx][attr] = 1.0
                vectorized_row  = vectorized_row + list(attr_dict[idx].values())
        vectorized_data.append(vectorized_row)
        reset_dict()

    return vectorized_data, vectorized_labels

def averaged_perceptron(train_set, train_labels, test_data, test_labels):
    w = np.zeros(len(train_set[0]))
    t = np.zeros(len(train_set[0]))

    for l in range(training_epochs):
        for i in range(len(train_set)):
            h = 0.0
            if np.dot(w, train_set[i]) >= 0:
                h = 1.0

            w = w + ((train_labels[i] - h) * train_set[i])
            t = t + w

    return 1.0/(training_epochs * len(train_set)) * t

def reset_dict():
    global attr_dict

    for key in attr_dict.keys():
        for i in attr_dict[key].keys():
            attr_dict[key][i] = 0.0

def predict(data_set, weights):
    predictions = []
    for row in data_set:
        activation = weights[0]
        for idx in range(len(row) - 1):
            activation += weights[idx + 1] * row[idx]

        if activation >= 0.0:
            predictions.append(1.0)
        else:
            predictions.append(0.0)

    return predictions

#computes the accuracy of classify_test_documents with correct results
def get_accuracy(predictions, accurate_classifications):
    inaccurate_count = 0

    for i in range(len(predictions)):
        if predictions[i] != accurate_classifications[i]:
            inaccurate_count += 1

    return 1.0 - (float(inaccurate_count) / len(predictions))

def main():
    build_attr_dict()
    training_set, test_set = read_data("game_attrdata_train.dat", "game_attrdata_test.dat")
    vectorized_data_train, vectorized_labels_train = vectorize_data(training_set)
    vectorized_data_test, vectorized_labels_test = vectorize_data(test_set)

    weights = averaged_perceptron(vectorized_data_train, vectorized_labels_train, vectorized_data_test, vectorized_labels_test)


if __name__ == "__main__":
    main()
