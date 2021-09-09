#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from loader import Dataset

#--------------------------------
# Create: Lightning Data Module
#--------------------------------

class PLDM(LightningDataModule):

    def __init__(self, params):

        super().__init__()

        # Load: Dataset Parameters

        self.data = params["train"]

        # Load: Processing Parameters

        self.batch = params["batch_size"]
        self.workers = params["num_workers"]

    #----------------------------
    # Create: Training Datasets 
    #----------------------------

    def setup(self, stage: Optional[str] = None):

        # Create: Pytorch Datasets

        self.train = Dataset(self.data)
        self.valid = Dataset(self.data)
        
    #----------------------------
    # Create: Training DataLoader
    #----------------------------

    def train_dataloader(self):

        return DataLoader( self.train, batch_size = self.batch, 
                           num_workers = self.workers, shuffle = 1, persistent_workers = 1 )

    #----------------------------
    # Create: Validation Loader
    #----------------------------

    def val_dataloader(self):

        return DataLoader( self.valid, batch_size = self.batch, 
                           num_workers = self.workers, persistent_workers = 1 )

