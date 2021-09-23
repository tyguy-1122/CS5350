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
