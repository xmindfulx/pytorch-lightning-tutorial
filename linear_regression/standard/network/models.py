#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import torch.nn as nn                                                                      

from pytorch_lightning import LightningModule

#--------------------------------
# Initialize: Lightining Model
#--------------------------------

class Linear_Regression(LightningModule):

    def __init__(self, params):

        super().__init__()

        # Load: Model Parameters
        
        self.max_epochs = params["num_epochs"]
        self.learning_rate = params["learning_rate"]

        # Initialize: Regression Model 

        self.regressor = nn.Linear(1, 1)

    #----------------------------
    # Create: Objective Function
    #----------------------------

    def objective(self, preds, labels):
    
        # Format: Labels

        labels = labels.type(preds.type())

        # Objective: Mean Squared Error

        cost = nn.MSELoss() 

        loss = cost(preds, labels) 

        # Logging: Loss

        self.log("loss", loss, on_step = True, on_epoch = True)

        return loss

    #----------------------------
    # Create: Optimizer Function
    #----------------------------

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

        return optimizer

    #----------------------------
    # Create: Model Forward Pass
    #----------------------------

    def forward(self, samples):

        return self.regressor(samples)

    #----------------------------
    # Create: Train Cycle (Epoch)
    #----------------------------

    def training_step(self, batch, batch_idx):

        # Load: Data Batch

        samples, labels = batch

        preds = self(samples)

        # Calculate: Training Loss

        loss = self.objective(preds, labels)
       
        return loss

    #----------------------------
    # Run: Post Training Script
    #----------------------------

    def training_epoch_end(self, train_step_outputs): 

        # Update: Training Plots

        if(self.current_epoch > 0):

            logger = self.logger.experiment

            logger.log_training_loss(self.current_epoch)
    
            # Finalize: Learned Features & Metrics ( Video )

            if(self.current_epoch == self.max_epochs - 1):
                
                logger.finalize_results()

    #----------------------------
    # Create: Validation Cycle 
    #----------------------------

    def validation_step(self, batch, batch_idx):

        samples, labels = batch

        preds = self(samples)
    
        return samples, labels, preds

    #----------------------------
    # Run: Post Validation Script
    #----------------------------

    def validation_epoch_end(self, val_step_outputs): 

        # Organize: Validation Outputs
 
        all_samples, all_labels, all_preds = [], [], []
    
        for group in val_step_outputs:

            samples, labels, preds = group

            all_labels.append( labels )
            all_samples.append( samples )
            all_preds.append( preds.detach() )

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_samples = torch.cat(all_samples)

        # Logger: Visualizations  

        logger = self.logger.experiment
        logger.log_linear_regression(all_samples, all_labels, all_preds, self.current_epoch)

