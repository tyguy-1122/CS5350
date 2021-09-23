from utilities import extract_data_from_csv
import math

def calc_entropy(data):
    '''
    Calculate the entropy of a data set using the following equation:
    Entropy = Σ(label_occurances/num_rows) * log2(label_occurances/num_rows)
    '''
    label_counts = {}
    for row in data:
        label = row[-1]
        if label in label_counts.keys():
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    num_labels = len(data)
    entropy = 0
    for key, value in label_counts:
        entropy += (value/num_labels) * math.log2(value/num_labels)
    
    return entropy
    
def calc_information_gain(data):
    '''
    Calculates the information gain for each attribute in the dataset using the following equation:
    IG = total_entropy - Σ(label_occurances/num_rows) * log2(label_occurances/num_rows)
    '''
    label_gains = {}
    # Split the data sets for each attribute
    for i in range(len(data) - 1):
        #####################
        #####################
        #####################
        # TODO: Pick up here Thursday

class DecisionTreeNode:
    def __init__(self, is_leaf, content, paths=[], children=[]):
        if len(paths) != len(children): raise ValueError('Must have the same number of paths and children')
        self.is_leaf = is_leaf
        self.content = content
        self.paths = paths
        self.children = children

class DecisionTree:
    def __init__(self, root_node):
        self.root = root_node

    def learn_id3(self, data_subset, parent_node=None):    
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
        
        else: # Find the attribute that best splits the data
            IG_arr = calc_information_gain(data)



