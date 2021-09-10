#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import torch.nn as nn                                                                      
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule

#--------------------------------
# Initialize: Lightining Model
#--------------------------------

class Linear_Regression(LightningModule):

    def __init__(self, params):

        super().__init__()

        # Load: Logger

        self.logger_choice = params["logger_choice"]

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

        if(self.logger_choice == 1):
            
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

        if(self.logger_choice == 0):
        
            self.log_linear_regression(all_samples, all_labels, all_preds, self.current_epoch)
            
        else:
            
            logger = self.logger.experiment
            logger.log_linear_regression(all_samples, all_labels, all_preds, self.current_epoch)

    #----------------------------
    # Logging: Feature Embeddings
    #----------------------------

    def log_linear_regression(self, samples, labels, preds, epoch, z = 4, f_s = 20, p_s = (15, 11)):

        # Format: Feature Vectors

        preds = torch.squeeze(preds)
        samples = torch.squeeze(samples)

        # Format: Plot

        plt.style.use("seaborn")

        # Assign: Figure Name

        name = str(epoch).zfill(z) + ".png"

        # Visualize: Results

        fig = plt.figure(figsize = p_s)

        ax = fig.add_subplot()

        ax.scatter( samples, labels, c = "blue")
        ax.plot( samples, preds, c = "red")
        ax.set_xlabel("x1", fontsize = f_s)
        ax.set_ylabel("x2", fontsize = f_s)

        fig.suptitle("Learned Linear Regression", fontsize = f_s)

        plt.subplots_adjust(top = 0.90)

        logger = self.logger.experiment
        logger.add_figure(name,  plt.gcf())
