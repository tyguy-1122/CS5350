import random
import copy

class Perceptron:
    def __init__(self, data, T, type='STANDARD'):
        self.type = type
        if type == 'STANDARD':
            self.create_standard_classifier(data, T)
        elif type == 'VOTED':
            self.create_voted_classifier(data, T)
        elif type == 'AVERAGE':
            self.create_average_classifier(data, T)
        else:
            raise Exception('Invalid type')

    def create_standard_classifier(self, data, T):
        self.classifier_history = []
        self.classifier = [0] * (len(data[0]) - 1)
        self.learning_rate = .01      # What should my learning rate be here???

        for i in range(T):
            self.classifier_history.append(self.classifier)
            random.shuffle(data)
            # Look at each example in training data
            for j in range(len(data)):
                prediction = self.make_prediction(self.classifier, data[j])
                true_label = data[j][-1]
                if prediction != true_label:
                    # Update weight vector on error
                    for k in range(len(self.classifier)):
                        if true_label == 0:
                            self.classifier[k] -= self.learning_rate * data[j][k]
                        else:
                            self.classifier[k] += self.learning_rate * data[j][k]

    def create_average_classifier(self, data, T):
        weight_vector = [0] * (len(data[0]) - 1)
        a = [0] * (len(data[0]) - 1) 
        self.learning_rate = .01
        self.classifier_history = [.deepcopy(weight_vectorcopy)]

        for i in range(T):
            random.shuffle(data)
            # Look at each example in training data
            for j in range(len(data)):
                prediction = self.make_prediction(weight_vector, data[j])
                true_label = data[j][-1]
                if prediction != true_label:
                    # Update weight vector on error
                    for k in range(len(weight_vector)):
                        if true_label == 0:
                            weight_vector[k] -= self.learning_rate * data[j][k]
                        else:
                            weight_vector[k] += self.learning_rate * data[j][k]
                    
                    self.classifier_history.append(copy.deepcopy(weight_vector))
                
                for k in range(len(a)):
                    a[k] += weight_vector[k]
        
        for i in range(len(a)):
            a[i] /= T * len(data)
        
        self.classifier = a

    def create_voted_classifier(self, data, T):
        self.classifier = []
        weight_vector = [0] * (len(data[0]) - 1)
        c = 1
        self.learning_rate = .01

        for i in range(T):
            random.shuffle(data)
            # Look at each example in training data
            for j in range(len(data)):
                prediction = self.make_prediction(weight_vector, data[j])
                true_label = data[j][-1]
                if prediction != true_label:
                    self.classifier.append((copy.deepcopy(weight_vector), c))
                    # Update weight vector on error
                    for k in range(len(weight_vector)):
                        if true_label == 0:
                            weight_vector[k] -= self.learning_rate * data[j][k]
                        else:
                            weight_vector[k] += self.learning_rate * data[j][k]
                        c = 1
                else:
                    c += 1

    def classify_data(self, row):
        if self.type == 'STANDARD' or self.type == 'AVERAGE':
            dot_prod = 0
            for i in range(len(self.classifier)):
                dot_prod += self.classifier[i] * row[i]
            
            return 0 if dot_prod < 0 else 1
        
        elif self.type == 'VOTED':
            votes = { 0:0, 1:0 }
            for classifier in self.classifier:
                dot_prod = 0
                for i in range(len(classifier[0])):
                    dot_prod += classifier[0][i] * row[i]
                
                if dot_prod < 0: votes[0] += 1 * classifier[1]
                else: votes[1] += 1 * classifier[1]
            return max(votes, key=votes.get)
    
    def make_prediction(self, weight_vector, row):
        dot_prod = 0
        for i in range(len(weight_vector)):
            dot_prod += weight_vector[i] * row[i]
        
        return 0 if dot_prod < 0 else 1