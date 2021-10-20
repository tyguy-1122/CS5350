from math import log, exp
from DecisionTree.decision_tree import DecisionTree

def calc_weighted_classification_error(predicted_labels, correct_labels, weights):
    '''
    Returns the classification error in percentage.
    '''
    if len(predicted_labels) != len(correct_labels):
        raise ValueError('The number of predicted labels should be equal to the number of correct labels!')
    
    total_error = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != correct_labels[i]:
            total_error += weights[i]
    
    return total_error

class AdaBoost:
    def __init__(self, data, possible_attribute_values, column_headers, T):
        self.column_headers = column_headers
        self.possible_attribute_values = possible_attribute_values
        self.weights = []
        self.weak_classifiers = []
        self.create_classifier(data, T)

    def create_classifier(self, data, T):
        for i in range(T):
            if i == 0: # Use 1/m for the first row weight
                for row in data:
                    row.insert(0, 1/len(data))
            else: # Update the values of the weights
                total_weight = 0
                multiplier = self.weights[-1]
                for j in range(len(data)):
                    if predicted_labels[j] == true_labels[j]:
                        data[j][0] *= exp(-1 * multiplier)
                    else:
                         data[j][0] *= exp(multiplier)
                    total_weight += data[j][0]
                for j in range(len(data)):
                    data[j][0] /= total_weight
           
            # Add the classifier to the weak_classifiers
            next_classifer = DecisionTree(data, self.possible_attribute_values, self.column_headers, max_depth=2, weighted_data=True, numerical_vals=True)
            self.weak_classifiers.append(next_classifer)
            
            # Compute the weight of the weak_classifier and add to the list of weights
            true_labels = [row[-1] for row in data]
            weights = [row[0] for row in data]
            predicted_labels = []
            for j in range(len(data)):
                predicted_labels.append(next_classifer.classify_data(data[j]))

            error = calc_weighted_classification_error(predicted_labels, true_labels, weights)
            weight = .5 * log((1 - error)/error)
            self.weights.append(weight)
    
    def classify_data(self, row, label_map):
        classification = 0
        for i in range(len(self.weak_classifiers)):
            classification += label_map[self.weak_classifiers[i].classify_data(row)] * self.weights[i]

        if classification >= 0:
            return label_map['positive']
        return label_map['negative']

