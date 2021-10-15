import math
from statistics import mode
from statistics import median

##################################
# Static Methods
##################################

def get_label_counts(data, weighted_data):
    '''
    Creates a dictionary with labels as keys and the frequency of the labels as values
    '''
    label_counts = {}
    for row in data:
        label = row[-1]
        if weighted_data:
            if label in label_counts.keys():
                label_counts[label] += row[0]
            else:
                label_counts[label] = row[0]
        else:
            if label in label_counts.keys():
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    return label_counts

def calc_entropy(data, weighted_data):
    '''
    Calculates the entropy of a data set using the following equation:
    Entropy = -Σ(label_occurances/num_rows) * log2(label_occurances/num_rows)
    '''
    label_counts = get_label_counts(data, weighted_data)

    num_labels = 0
    if weighted_data:
        for i in range(len(data)):
            num_labels += data[i][0]
    else:
        num_labels = len(data)
    entropy = 0
    for key, value in label_counts.items():
        entropy -= (value/num_labels) * math.log2(value/num_labels)
    
    return entropy

def calc_majority_error(data, weighted_data):
    '''
    Calculate the majority error of a data set using the following equation:
    Majority Error = (num_non_majority_labels) / (total_rows)
    '''
    if weighted_data:
        raise ValueError('Majority Error not yet compatible with weighted data')
    label_counts = get_label_counts(data, weighted_data)
    
    num_labels = len(data)
    labels = [row[-1] for row in data]
    most_common_label = mode(labels)
    most_common_label_frequency = labels.count(most_common_label)
    majority_error = (len(labels) - most_common_label_frequency) / len(labels)

    return majority_error

def calc_gini_index(data, weighted_data):
    '''
    Calculate the gini index of a data set using the following equation:
    Gini Index = 1 - Σ(label_occurances/num_rows)^2
    '''
    if weighted_data:
        raise ValueError('Gini Index not yet compatible with weighted data')
    label_counts = get_label_counts(data, weighted_data)
    
    num_labels = len(data)
    gini = 1
    for key, value in label_counts.items():
        gini -= (value/num_labels) ** 2
    
    return gini

def calc_information_gain(data, remaining_attributes, method, weighted_data):
    '''
    Calculates the information gain for each attribute in the dataset using the following equations:

    1. Entropy = - Σ(label_occurances/num_rows) * log2(label_occurances/num_rows)
    2. Gini Index = 1 - Σ(label_occurances/num_rows)^2
    3. Majority Error = (num_non_majority_labels) / (total_rows)
    '''
    attribute_gains = {}
    for i in remaining_attributes:

        # Split the data sets for each attribute
        data_per_attribute_value = {}
        for row in data:
            if row[i] not in data_per_attribute_value:
                data_per_attribute_value[row[i]] = []
            data_per_attribute_value[row[i]].append(row)        
        
        gain = 0
        # Use the specified variant
        if method == 'ENT':
            current_entropy = calc_entropy(data, weighted_data)
            # Calculate entropy for each attribute value and subract from current entropy
            entropy_gain = current_entropy
            for key, value in data_per_attribute_value.items():
                numerator = 0
                denominator = len(data)
                if weighted_data:
                    denominator = 0
                    for j in range(len(value)):
                        numerator += value[j][0]
                    for j in range(len(data)):
                        denominator += data[j][0]
                else:
                    numerator = len(value)
                entropy_gain -= (numerator/denominator) * calc_entropy(value, weighted_data)
            gain = entropy_gain
        elif method == 'ME':
            current_me = calc_majority_error(data, weighted_data)
            # Calculate majority error for each attribute value and subract from current entropy
            me_gain = current_me
            for key, value in data_per_attribute_value.items():
                numerator = 0
                denominator = len(data)
                if weighted_data:
                    denominator = 0
                    for j in range(len(value)):
                        numerator += value[j][0]
                    for j in range(len(data)):
                        denominator += data[j][0]
                else:
                    numerator = len(value)       
                me_gain -= (numerator/denominator) * calc_majority_error(value, weighted_data)
            gain = me_gain
        elif method =='GINI':
            current_gini = calc_gini_index(data, weighted_data)
            # Calculate gini index for each attribute value and subract from current entropy
            gini_gain = current_gini
            for key, value in data_per_attribute_value.items():
                numerator = 0
                denominator = len(data)
                if weighted_data:
                    denominator = 0
                    for j in range(len(value)):
                        numerator += value[j][0]
                    for j in range(len(data)):
                        denominator += data[j][0]
                else:
                    numerator = len(value)                
                gini_gain -= (numerator/denominator) * calc_gini_index(value, weighted_data)
            gain = gini_gain
        else:
            raise TypeError('Invalid method type. Valid types are "ENT" (Entropy), "ME" (Majority Error), and "GINI" (GINI index)')

        # Add entropy to the gains dictionary
        attribute_gains[i] = gain
    
    return attribute_gains

################################
# Classses
################################

class DecisionTreeNode:
    '''
    Represents a node in the DecisionTree class
    '''
    def __init__(self, is_leaf, content):
        self.is_leaf = is_leaf
        self.content = content
        self.paths = []
        self.children = []

class DecisionTree:
    '''
    Represents a decision tree structure used for classification. The tree is trained using the ID3 
    decision tree learning algorithm.
    '''
    def __init__(self, data, possible_attribute_values, column_headers, max_depth=-1, method='ENT',
     unknown_vals=False, numerical_vals=False, weighted_data=False, random_attributes=None):
        self.column_headers = column_headers
        self.max_depth = max_depth
        self.possible_attribute_values = possible_attribute_values
        self.weighted_data = weighted_data

        remaining_attributes = []
        if random_attributes != None:
            remaining_attributes = random_attributes
        elif weighted_data:
            remaining_attributes = [i for i in range(1, len(possible_attribute_values) + 1)]
        else:
            remaining_attributes = [i for i in range(len(possible_attribute_values))]

        # Preprocess the data if necessary
        if numerical_vals:
            self.convert_numerical_to_binary(data) # Might need to handle unkown cases in finding median
        if unknown_vals:
            self.handle_missing_attributes(data)

        # Learn the tree
        self.learn_id3(data, self.max_depth, remaining_attributes, method)
    
    def convert_numerical_to_binary(self, data):
        for i in range(len(data[0]) - 1):
            if self.weighted_data and i == 0: continue
            # Determine if a column is numerical
            if ['-', '+'] == self.possible_attribute_values[self.column_headers[i]] and data[0][i] not in ['-', '+']: # Column is numerical and not adjusted
                # Find the median value
                median_val = median([int(row[i]) for row in data])
                # Replace all numbers with either + or -
                for row in data:
                    if int(row[i]) >= median_val:
                        row[i] = '+'
                    else:
                        row[i] = '-'
        return data

    def handle_missing_attributes(self, data):
        # Find the most common attribute value for each row
        most_common_attributes = []
        for i in range(len(data[0]) - 1):
            most_common_attributes.append(mode([row[i] for row in data if row[i] != 'unknown']))
        # Replace unknown attributes with most common attribute value 
        for j in range(len(data)):
            for i in range(len(data[0]) - 1):
                if data[j][i] == 'unknown':
                    data[j][i] = most_common_attributes[i]
        return data

    def classify_data(self, row):
        if len(self.column_headers) != len(row):
            raise ValueError('Row does not match the data set this tree was trained on!')
  
        curr_node = self.root
        while not curr_node.is_leaf: # Traverse the tree until we reach a leaf node
            column_index = self.column_headers.index(curr_node.content)
            path = row[column_index]
            child_node_index = curr_node.paths.index(path)
            curr_node = curr_node.children[child_node_index]
        return curr_node.content # Return the label (content of the leaf node)

    def learn_id3(self, data_subset, max_depth, remaining_attributes, method, parent_node=None): 
        '''
        Creates a decision tree using the ID3 algorithm.

        Valid method types are "ENT" (Entropy), "ME" (Majority Error), and "GINI" (GINI index)
        '''   

        # Find the most common label in the data
        most_common_label = ''
        if not self.weighted_data:
            labels = [row[-1] for row in data_subset]
            most_common_label = mode(labels)
        else:
            most_common_label_dict = {}
            for row in data_subset:
                if row[-1] not in most_common_label_dict:
                    most_common_label_dict[row[-1]] = 0
                most_common_label_dict[row[-1]] += row[0]
            most_common_label = max(most_common_label_dict, key=most_common_label_dict.get)

        # Base Case: All rows have the same label
        first_label = data_subset[0][-1]
        all_labels_equal = True
        for row in data_subset:
            if row[-1] != first_label:
                all_labels_equal = False
                break

        if all_labels_equal: # Create a leaf node
            if parent_node == None:
                self.root = DecisionTreeNode(True, first_label)
            else:    
                parent_node.children.append(DecisionTreeNode(True, first_label))
        
        # Base Case: Their are no more attributes left to split on or max_depth == 0 -> make leaf with most common value
        elif len(remaining_attributes) == 0 or max_depth == 0:
            parent_node.children.append(DecisionTreeNode(True, most_common_label))
        
        else: 
            # Find the attribute that best splits the data
            IG_dict = calc_information_gain(data_subset, remaining_attributes, method, self.weighted_data)
            next_node_index = max(IG_dict, key=IG_dict.get)
            current_node = DecisionTreeNode(False, self.column_headers[next_node_index])

            # Add the node to the tree
            if parent_node == None:
                self.root = current_node
            else:
                parent_node.children.append(current_node)

            # Recurse for each attribute of the current node
            for attribute_value in self.possible_attribute_values[current_node.content]:
                # Split the data subset for each attribute path
                attribute_data = []
                for row in data_subset:
                    if row[next_node_index] == attribute_value:
                        attribute_data.append(row)
                
                current_node.paths.append(attribute_value) # Append the path

                # Base Case: There is no data for an attribute_value -> make leaf with most common value
                if not attribute_data:
                    current_node.children.append(DecisionTreeNode(True, most_common_label))
                else:
                    remaining_attributes_copy = remaining_attributes.copy()
                    del remaining_attributes_copy[remaining_attributes_copy.index(next_node_index)]
                    self.learn_id3(attribute_data, max_depth - 1, remaining_attributes_copy, method, parent_node=current_node)
