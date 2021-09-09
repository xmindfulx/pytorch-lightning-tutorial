#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch                                                                                
import torch.utils.data as tech 

#--------------------------------
# Initialize: Custom CAN Dataset
#--------------------------------

class Dataset(tech.Dataset):
    
    def __init__(self, data):

        self.data = data

        # Create: Dataset 

        self.make_dataset()

    #----------------------------
    # Populate: Training Dataset
    #----------------------------

    def make_dataset(self):
   
        # Load: Samples, Labels

        all_labels, all_samples = self.data.labels, self.data.samples

        # Create: Train Dataset
      
        self.data = [ [torch.tensor(sample), torch.tensor(label)] 
                      for sample, label in zip(all_samples, all_labels) ] 

    #----------------------------
    # Gather: Nth Sample, Dataset
    #----------------------------

    def __getitem__(self, index):
      
        # Gather: Samples
 
        samples = self.data[index][0]

        # Format: Samples

        samples = samples.float()

        # Replace: Samples

        self.data[index][0] = samples

        return self.data[index] 

    #---------------------------------
    # Gather: Number Dataset Samples
    #---------------------------------

    def __len__(self):
        
        return len(self.data) 

