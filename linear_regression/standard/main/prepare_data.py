#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import numpy as np

#--------------------------------
# Initialize: Custom Dataset 
#--------------------------------

class Dataset:

    def __init__(self, samples, labels):

        self.labels = labels
        self.samples = samples

#--------------------------------
# Load: Training Dataset (.CSV)
#--------------------------------

def load_data(path):

    data_file = open(path, "r")

    data = []
    for line in data_file:
        data.append([ float(ele.strip("\n")) for ele in line.split(",") ])

    data = np.asarray(data)
    
    samples, labels = data[:, :-1], data[:, -1]

    return Dataset(samples, labels) 

