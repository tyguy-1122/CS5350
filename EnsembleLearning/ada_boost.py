from math import log, exp
from DecisionTree.utilities import extract_data_from_csv
from DecisionTree.utilities import calc_classification_error
from DecisionTree.decision_tree import DecisionTree


class AdaBoost:
    def __init__(self, data, possible_attribute_values, column_headers, T):
        self.column_headers = column_headers
        self.possible_attribute_values = possible_attribute_values
        self.weights = []
        self.weak_classifiers = []
        self.create_classifier(data, T)

    def create_classifier(self, data, T):
        for i in range(T):
            if len(self.weak_classifiers) == 0: # Use 1/m for the first row weight
                for row in data:
                    row.insert(0, 1/len(data))
            else: # Update the values of the weights
                total_weight = 0
                for j in range(len(data)):
                    if predicted_labels[j] == true_labels[j]:
                        data[j][0] = data[j][0] * exp(-1 * self.weights[-1])
                    else:
                         data[j][0] = data[j][0] * exp(self.weights[-1])
                    total_weight += data[j][0]
                for j in range(len(data)):
                    data[j][0] /= total_weight
           
            # Add the classifier to the weak_classifiers
            next_classifer = DecisionTree(data, self.possible_attribute_values, self.column_headers, max_depth=2, weighted_data=True, numerical_vals=True)
            self.weak_classifiers.append(next_classifer)
            
            # Compute the weight of the weak_classifier and add to the list of weights
            true_labels = [row[-1] for row in data]
            predicted_labels = []
            for j in range(len(data)):
                predicted_labels.append(next_classifer.classify_data(data[j]))

            error = calc_classification_error(predicted_labels, true_labels) / 100
            weight = .5 * log((1 - error)/error)
            self.weights.append(weight)
    
    def classify_data(self, row):
        for classifier in self.weak_classifiers:
            pass

    
if __name__ == '__main__':
    bank_data_training = extract_data_from_csv('/Users/tylerjones/Documents/CS5350/CS5350/DecisionTree/bank/train.csv')

    possible_attribute_values_bank = {
    'age': ['-','+'],
    'job': ['admin.','unemployed','management','housemaid','entrepreneur','student',
        'blue-collar','self-employed','retired','technician','services'],
    'marital' : ['married','divorced','single'],
    'education': ['secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['-','+'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['telephone','cellular'],
    'day': ['-','+'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['-','+'],
    'campaign': ['-','+'],
    'pdays': ['-','+'],
    'previous': ['-','+'],
    'poutcome': ['other','failure','success']
    }

    possible_attribute_values_bank_with_unknown = {
    'age': ['-','+'],
    'job': ['admin.','unknown','unemployed','management','housemaid','entrepreneur','student',
        'blue-collar','self-employed','retired','technician','services'],
    'marital' : ['married','divorced','single'],
    'education': ['unknown','secondary','primary','tertiary'],
    'default': ['yes','no'],
    'balance': ['-','+'],
    'housing': ['yes','no'],
    'loan': ['yes','no'],
    'contact': ['unknown','telephone','cellular'],
    'day': ['-','+'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['-','+'],
    'campaign': ['-','+'],
    'pdays': ['-','+'],
    'previous': ['-','+'],
    'poutcome': ['unknown','other','failure','success']
    }

    classifier = AdaBoost(bank_data_training, possible_attribute_values_bank,
     ['weights', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
        'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'], 2)

