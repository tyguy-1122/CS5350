This is a machine learning library developed by Tyler Jones for CS5350 at the University of Utah

How to use DecisionTree:
1. Create a DecisionTree object
    Parameters:
    - data: The training data set
    - possible_attribute_values: a diction of indexes mapped to a list of the possible attribute values of
    the column at that index
    - column_headers: A list of the names of the features in order
    - max_depth=1: The maximum depth of the tree
    - method='ENT': 'ENT'->Entropy, 'ME'->Majority Error, or 'GINI'->Gini index
    - unknown_vals=False: Whether or not to replace instances of 'unknown' in the data set
    with the most common label
    - numerical_vals=False: Whether or not to replace numerical attributes with a boolean value indicating
    that that value is less than or greater than the mean

2. Classify Data
    Call the classify_data function of the DecisionTree object created in the previous step
    Parameters:
    - row: the row of data in the same order as the training set