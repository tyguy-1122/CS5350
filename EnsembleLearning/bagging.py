from DecisionTree.decision_tree import DecisionTree
from random import randrange
from statistics import median

def random_sample_data(data, num_samples):
    random_sample = []
    for i in range(len(data)):
        random_sample.append(data[randrange(len(data))])
    return random_sample

class Bagging:
    def __init__(self, data, possible_attribute_values, column_headers, num_weak_classifiers):
        self.weak_classifiers = []
        self.create_classifier(data, possible_attribute_values, column_headers, num_weak_classifiers)
    
    def create_classifier(self, data,  possible_attribute_values, column_headers, num_weak_classifiers):
        # Convert data to binary from numerical
        for i in range(len(data[0]) - 1):
            if ['-', '+'] == possible_attribute_values[column_headers[i]] and data[0][i] not in ['-', '+']: # Column is numerical and not adjusted
                # Find the median value
                median_val = median([int(row[i]) for row in data])
                # Replace all numbers with either + or -
                for row in data:
                    if int(row[i]) >= median_val:
                        row[i] = '+'
                    else:
                        row[i] = '-'
        for i in range(num_weak_classifiers):
            random_sample = random_sample_data(data, len(data))
            weak_classifier = DecisionTree(random_sample, possible_attribute_values, column_headers)
            self.weak_classifiers.append(weak_classifier)
    
    def classify_data(self, row):
        votes = {}
        for weak_classifier in self.weak_classifiers:
            vote = weak_classifier.classify_data(row)
            if vote not in votes:
                votes[vote] = 0
            votes[vote] += 1

        return max(votes, key=votes.get)