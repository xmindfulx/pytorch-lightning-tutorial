#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import numpy as np
import torch.nn as nn                                                                      
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule

#--------------------------------
# Initialize: Lightining Model
#--------------------------------

class MLP(LightningModule):

    def __init__(self, params):

        super().__init__()

        # Load: Logger
        
        self.logger_choice = params["logger_choice"]        

        # Load: Model Parameters
        
        self.max_epochs = params["num_epochs"]
        self.learning_rate = params["learning_rate"]

        # Initialize: MLP Model 

        self.evaluate = nn.Sequential( nn.Linear(2, 5),
                                       nn.ReLU(),
                                       nn.Linear(5, 10),
                                       nn.ReLU(),
                                       nn.Linear(10, 2) )

    #----------------------------
    # Create: Objective Function
    #----------------------------

    def objective(self, preds, labels):
    
        # Format: Labels

        labels = labels.long()

        # Objective: Mean Squared Error

        cost = nn.CrossEntropyLoss() 

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

        return self.evaluate(samples)

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

        # Calculate: Performance

        results = self.calculate_performance(all_preds, all_labels)

        # Logger: Performance

        self.log("recall", results["model_recall"], on_epoch = True)
        self.log("accuracy", results["model_accuracy"], on_epoch = True)
        self.log("precision", results["model_precision"], on_epoch = True)

        preds, feature_space = self.prepare_world(all_samples)
            
        if(self.current_epoch > 0):
            
            if(self.logger_choice == 0):
                
                # Logger: Feature Visualizations  
                
                self.log_features(all_samples, all_labels, feature_space, preds, self.current_epoch)
                
            else:
                    
                logger = self.logger.experiment

                # Logger: Validation Plots

                logger.log_valid_results(self.current_epoch)

                # Logger: Feature Visualizations  

                logger.log_features(all_samples, all_labels, feature_space, preds, self.current_epoch)

    #----------------------------
    # Run: Validation Metrics
    #----------------------------

    def calculate_performance(self, all_preds, all_labels):

        unique_labels = np.unique(all_labels.numpy())

        confusion_matrix = np.zeros( [len(unique_labels), len(unique_labels)] )

        for pred, label in zip(all_preds, all_labels):

            label = int(label)

            prediction = np.argmax(pred.numpy())
            
            confusion_matrix[label][prediction] += 1                
               
        confusion_matrix = confusion_matrix.astype(int)        
     
        return self.calculate_statistics(confusion_matrix)

    #----------------------------
    # Calculate: Basic Statistics
    #----------------------------

    def calculate_statistics(self, matrix):
        
        results = {}
        all_precision, all_recall, all_fscore, all_accuracy = [], [], [], []
        
        for target_class in range(matrix.shape[0]):
            
            true_positive = matrix[target_class, target_class]
            false_negatives = np.sum(matrix[target_class, :] ) - true_positive
            false_positives = np.sum(matrix[:, target_class] ) - true_positive

            if(true_positive != 0):            
                precision = true_positive / ( true_positive + false_positives ) 
                recall = true_positive / ( true_positive + false_negatives ) 
                fscore = ( 2 * recall * precision ) / ( recall + precision )
                accuracy = true_positive / np.sum(matrix[target_class, : ])
            else:
                precision = recall = fscore = accuracy = 0

            all_precision.append(np.round(precision, 3))
            all_accuracy.append(np.round(accuracy, 3))
            all_fscore.append(np.round(fscore, 3))
            all_recall.append(np.round(recall, 3))

        results['model_precision'] = np.round(np.mean(all_precision), 3)
        results['model_accuracy'] = np.round(np.mean(all_accuracy), 3)
        results['model_recall'] = np.round(np.mean(all_recall), 3)
        results['model_fscore'] = np.round(np.mean(all_fscore), 3)

        results['class_precision'] = all_precision
        results['class_accuracy'] = all_accuracy
        results['class_recall'] = all_recall
        results['class_fscore'] = all_fscore 
        
        results['confusion'] = matrix
        
        return results 

    #----------------------------
    # Generation: Feature Space
    #----------------------------
    
    def prepare_world(self, samples, offset = 0.5, precision = 0.05):

        # Gather: Mins, Maxes Dataset ( Adjust Offset )

        y_min = torch.min(samples[:, 0]) - offset
        y_max = torch.max(samples[:, 0]) + offset
        x_min = torch.min(samples[:, 1]) - offset
        x_max = torch.max(samples[:, 1]) + offset

        # Create: 2D Feature Space 
    
        y_vals = torch.arange(y_min, y_max, precision)
        x_vals = torch.arange(x_min, x_max, precision)

        all_points = [ [y, x] for y in y_vals for x in x_vals ]

        all_points = torch.tensor(all_points)

        # Evaluate: 2D Feature Space ( MLP )

        predictions = []
        for sample in all_points:
            
            sample = torch.unsqueeze(sample, dim = 0)
            predictions.append( torch.argmax(self(sample).detach()) )
            
        return torch.tensor(predictions), all_points

    #----------------------------
    # Logging: Feature Embeddings
    #----------------------------

    def log_features(self, samples, labels, feature_space, preds, epoch, z = 4, f_s = 20, p_s = (15, 11)):

        # Assign: Figure Name

        name = str(epoch).zfill(z) + ".png"

        # Format: Plot

        plt.style.use("seaborn")

        # Assign: Colors

        face_colors = [ "blue" if(ele == 0) else "red" for ele in labels ]
        back_colors = [ "darkblue" if(ele == 0) else "darkred" for ele in preds ]

        # Plot: Dataset & Feature Space

        fig, ax = plt.subplots(figsize = p_s)

        ax.scatter( feature_space[:, 0], feature_space[:, 1], c = back_colors )
        ax.scatter( samples[:, 0], samples[:, 1], s = 200, 
                    linewidths = 3, edgecolor = "black", c = face_colors )

        ax.set_xlabel("x1", fontsize = f_s)
        ax.set_ylabel("x2", fontsize = f_s)

        fig.suptitle("Learned Decision Boundary", fontsize = f_s)

        plt.subplots_adjust(top = 0.90)
      
        logger = self.logger.experiment
        logger.add_figure(name,  plt.gcf())

