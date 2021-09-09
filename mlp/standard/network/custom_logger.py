#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import cv2
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment

#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Writer:

    def __init__(self, path, name = "metrics.csv"):

        self.name = name
        self.path = path

        # Initialize: Metrics

        self.metrics = []

        # Initialize: Paths ( Images, Metrics )

        self.path_metrics = os.path.join(self.path, self.name)

        self.path_videos = os.path.join(self.path, "videos")
        self.path_features = os.path.join(self.path, "features")
        self.path_validation = os.path.join(self.path, "validation")
        self.path_training_loss = os.path.join(self.path, "training_loss")

        if(os.path.exists(self.path)):
    
            shutil.rmtree(self.path)

        os.makedirs(self.path_videos)
        os.makedirs(self.path_features)
        os.makedirs(self.path_validation)
        os.makedirs(self.path_training_loss)

    #----------------------------
    # Logging: Numeric Results
    #----------------------------

    def log_metrics(self, metrics_dict, step = None):
    
        if step is None:
            step = len(self.metrics)

        self.metrics.append(metrics_dict)

    #----------------------------
    # Logging: Training Progress
    #----------------------------

    def log_training_loss(self, epoch, z = 4, f_s = 20, p_s = (15, 11)):

        # Format: Plot

        plt.style.use("seaborn")

        # Update: Save Path

        name = str(epoch).zfill(z) + ".png"
        path_save = os.path.join(self.path_training_loss, name)
    
        # Load: Current Progress

        headers, info, empty = self.load_csv(self.path_metrics)

        # Gather: Metric Indices

        i_e = np.where(headers == "epoch")
        i_l = np.where(headers == "loss_epoch")

        # Gather: Metric Values

        loss = info[:, i_l]
        epochs = info[:, i_e]

        # Remove: Empty Values

        indices = np.where(loss != empty)

        loss = loss[indices]
        epochs = epochs[indices]
    
        # Display: Results

        fig, ax = plt.subplots(figsize = p_s)  

        ax.plot(epochs, loss, "-o")
        ax.set_xlabel("epoch", fontsize = f_s)
        ax.set_ylabel("loss", fontsize = f_s)

        fig.suptitle("Training Loss", fontsize = f_s)

        plt.subplots_adjust(top = 0.90)
        plt.savefig(path_save)
        plt.close()

    #----------------------------
    # Logging: Validation Results
    #----------------------------

    def log_valid_results(self, epoch, z = 4, f_s = 20, p_s = (15, 11)):

        # Format: Plot

        plt.style.use("seaborn")

        # Update: Save Path

        name = str(epoch).zfill(z) + ".png"
        path_save = os.path.join(self.path_validation, name)
    
        # Load: Current Progress

        headers, info, empty = self.load_csv(self.path_metrics)

        # Gather: Metric Indices

        i_e = np.where(headers == "epoch")
        i_r = np.where(headers == "recall")
        i_a = np.where(headers == "accuracy")
        i_p = np.where(headers == "precision")

        # Gather: Metric Values

        epochs = info[:, i_e]
        recall = info[:, i_r]
        accuracy = info[:, i_a]
        precision = info[:, i_p]

        # Remove: Empty Values

        indices = np.where(accuracy != empty)

        epochs = epochs[indices]
        recall = recall[indices]
        accuracy = accuracy[indices]
        precision = precision[indices]
 
        # Display: Results

        fig, ax = plt.subplots(figsize = p_s)  

        ax.plot(epochs, recall, "-o", label = "recall")
        ax.plot(epochs, accuracy, "-o", label = "accuracy")
        ax.plot(epochs, precision, "-o", label = "precision")

        ax.set_xlabel("epoch", fontsize = f_s)
        ax.set_ylabel("performance", fontsize = f_s)

        ax.legend()

        fig.suptitle("Validation Performance", fontsize = f_s)

        plt.subplots_adjust(top = 0.90)
        plt.savefig(path_save)
        plt.close()

    #----------------------------
    # Logging: Feature Embeddings
    #----------------------------

    def log_features(self, samples, labels, feature_space, preds, epoch, z = 4, f_s = 20, p_s = (15, 11)):

        # Update: Save Path

        name = str(epoch).zfill(z) + ".png"
        path_save = os.path.join(self.path_features, name)

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
        plt.savefig(path_save)
        plt.close()
       
    #----------------------------
    # Update: Performance Metrics 
    #----------------------------

    def save(self):

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(self.path_metrics, "w", newline="") as f:
            self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
            self.writer.writeheader()
            self.writer.writerows(self.metrics)

    #----------------------------
    # Finalize: Learned Metrics
    #----------------------------

    def finalize_results(self):

        # Format: Saved Features ( Video )
   
        path_video = os.path.join(self.path_videos, "features.mp4")

        self.make_video( self.path_features, path_video )

        # Format: Training Loss ( Video )

        path_video = os.path.join(self.path_videos, "training_loss.mp4")

        self.make_video( self.path_training_loss, path_video )

        # Format: Validation Results ( Video )

        path_video = os.path.join(self.path_videos, "validation_results.mp4")

        self.make_video( self.path_validation, path_video )

    #----------------------------
    # Load: Performance Metrics 
    #----------------------------

    def load_csv(self, path, delimiter = ",", empty = -99999):
        
        data_file = open(path, "r")

        info = []

        for i, line in enumerate(data_file):
        
            line = [ ele.strip("\n") for ele in line.split(delimiter) ]

            if(i == 0): 

                headers = line

            else:

                line = [ float(ele) if( ele != "") else empty for ele in line ]

                info.append( line )

        data_file.close()
        
        return np.asarray(headers), np.asarray(info), empty

    #----------------------------
    # Save: Model Metrics (Video)
    #----------------------------

    def make_video(self, path_images, path_save, fps = 3, file_types = [".png", ".jpg"]): 

        # Gather: Relevant Files ( Sorted )

        all_files = [ ele for ele in os.listdir(path_images) if(ele[-4:] in file_types) ]

        all_files.sort()

        # Gather: Image Dimensions

        frame = cv2.imread(os.path.join(path_images, all_files[0]))
        
        height, width, _ = frame.shape 

        # Create: Video ( MP4 )

        video = cv2.VideoWriter(path_save, 0x7634706d, fps, (width,height))

        for current_file in all_files:
        
            video.write( cv2.imread(os.path.join(path_images, current_file)) )

        cv2.destroyAllWindows()

        video.release()


#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Logger(LightningLoggerBase):

    def __init__( self, save_dir, name = "default", version = None, prefix = "" ):

        super().__init__()

        self._name = name 
        self._prefix = prefix
        self._experiment = None
        self._version = version
        self._save_dir = save_dir

    #----------------------------
    # Gather: Path (Root Folder)
    #----------------------------
    
    @property
    def root_dir(self) -> str:

        if not self.name:
            return self.save_dir

        return os.path.join(self.save_dir, self.name)

    #----------------------------
    # Gather: Path (Log Folder)
    #----------------------------

    @property
    def log_dir(self):

        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)

        return log_dir

    #----------------------------
    # Gather: Path (Save Folder)
    #----------------------------

    @property
    def save_dir(self):

        return self._save_dir

    #----------------------------
    # Gather: Experiment Version
    #----------------------------

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    #----------------------------
    # Gather: Experiment Title
    #----------------------------

    @property
    def name(self):
        return self._name

    #----------------------------
    # Log: Performance Metrics
    #----------------------------

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = self._add_prefix(metrics)
        self.experiment.log_metrics(metrics, step)

    #----------------------------
    # Log: Model Hyperparameters
    #----------------------------

    @rank_zero_only
    def log_hyperparams(self, params):

        pass

    #----------------------------
    # Initialize: Saver Actions
    #----------------------------

    @rank_zero_only
    def save(self):

        super().save()
        self.experiment.save()

    #----------------------------
    # Run: Post Training Code
    #----------------------------

    @rank_zero_only
    def finalize(self, status):

        self.save()

    #----------------------------
    # Run: Logger-Writer Object
    #----------------------------

    @property
    @rank_zero_experiment
    def experiment(self):

        if(self._experiment):

            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)

        self._experiment = Writer(path = self.log_dir)

        return self._experiment

