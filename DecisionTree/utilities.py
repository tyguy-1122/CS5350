def extract_data_from_csv(csv_file_path):
    '''
    Extract attributes and labels from a csv.
    Assumptions:
    - Label is in the last column
    - Attributes and label are comma deliminated
    '''
    data = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data

def calc_classification_error(predicted_labels, correct_labels):
    '''
    Returns the classification error in percentage.
    '''
    if len(predicted_labels) != len(correct_labels):
        raise ValueError('The number of predicted labels should be equal to the number of correct labels!')
    
    count_incorrect = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != correct_labels[i]: count_incorrect += 1
    
    return (count_incorrect / len(predicted_labels)) * 100


