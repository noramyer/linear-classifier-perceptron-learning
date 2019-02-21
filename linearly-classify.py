#
# CSE 5522: HW 3 Linear Classification w/ Perceptron Learning
# Author: Nora Myer
#

from numpy import *
from matplotlib.pyplot import *
import json

data_vectorization_labels = [1.0, "is_weekday", "is_Saturday", "is_Sunday", "is_morning", "is_afternoon", "is_evening", "is_<30", "is_30-60", "is_>60", "is_silly", "is_happy", "is_tired", "friendsVisiting", "kidsPlaying", "atHome", "snacks"]
training_epochs = 30
attr_dict = {}
training_accuracies_current = []
test_accuracies_current = []
training_accuracies_averaged = []
test_accuracies_averaged = []
block_with_plot = True

def plot_assignments():
    plot(range(training_epochs), training_accuracies_current, 'bo-', label = 'Current train')
    plot(range(training_epochs), training_accuracies_averaged, 'b^-', label = 'Averaged train')
    plot(range(training_epochs), test_accuracies_current, 'ro-', label = 'Current test')
    plot(range(training_epochs), test_accuracies_averaged, 'r^-', label = 'Averaged test')

    legend()
    ylabel('Accuracy')
    xlabel('Epoch')
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
    train = np.array(train_set)
    w = np.zeros(len(train_set[0]))
    t = np.zeros(len(train_set[0]))

    for l in range(training_epochs):
        for i in range(len(train)):
            h = 0.0
            if np.dot(w, train[i]) >= 0.0:
                h = 1.0

            w = w + ((train_labels[i] - h) * train[i])
            t = t + w

        #get new accuracies
        print("****** At end of epoch: " + str(l) + " ******")
        current_and_averaged_model_accuracy(train_set, test_data, train_labels, test_labels, 1.0/((l+1) * len(train)) * t, w)
        print()

    return 1.0/(training_epochs * len(train)) * t

def current_and_averaged_model_accuracy(train_set, test_set, train_labels, test_labels, avg_weights, current_weights):
    global training_accuracies_averaged
    global training_accuracies_current
    global test_accuracies_averaged
    global test_accuracies_current

    train_p_current = predict(train_set, current_weights)
    train_a_current = get_accuracy(train_p_current, train_labels)

    test_p_current = predict(test_set, current_weights)
    test_a_current = get_accuracy(test_p_current, test_labels)

    print("Train, test accuracies on current model: " + str(train_a_current) + "  " + str(test_a_current))

    train_p_avg = predict(train_set, avg_weights)
    train_a_avg = get_accuracy(train_p_avg, train_labels)

    test_p_avg = predict(test_set, avg_weights)
    test_a_avg = get_accuracy(test_p_avg, test_labels)

    print("Train, test accuracies on averaged model: " + str(train_a_avg) + "  " + str(test_a_avg))
    training_accuracies_current.append(train_a_current)
    training_accuracies_averaged.append(train_a_avg)
    test_accuracies_current.append(test_a_current)
    test_accuracies_averaged.append(test_a_avg)

def reset_dict():
    global attr_dict

    for key in attr_dict.keys():
        for i in attr_dict[key].keys():
            attr_dict[key][i] = 0.0

def predict(data_set, weights):
    predictions = []
    for row in data_set:
        activation = 0.0
        for idx in range(len(row)):
            activation += weights[idx] * row[idx]

        if activation >= 1.0:
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
    print("Beginning linear classification .... \n")
    build_attr_dict()
    training_set, test_set = read_data("game_attrdata_train.dat", "game_attrdata_test.dat")
    vectorized_data_train, vectorized_labels_train = vectorize_data(training_set)
    vectorized_data_test, vectorized_labels_test = vectorize_data(test_set)

    avg_weights = averaged_perceptron(vectorized_data_train, vectorized_labels_train, vectorized_data_test, vectorized_labels_test)
    plot_assignments()
    print(avg_weights)

if __name__ == "__main__":
    main()
