#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from models import MLP
from data_module import PLDM
from custom_logger import Logger
from prepare_data import load_data

#--------------------------------
# Initialize: Training Pipeline
#--------------------------------

def run(params):

    seed = params["seed"] 
    use_gpu = params["use_gpu"]
    gpu_list = params["gpu_list"]
    valid_rate = params["valid_rate"]
    num_epochs = params["num_epochs"]
    path_dataset = params["path_dataset"]
    path_results = params["path_results"]
    logger_choice = params["logger_choice"]

    # Initialize: Gloabl Seed

    seed_everything(seed, workers = True)

    # Generate: Synthetic Dataset
     
    dataset = load_data(path_dataset)

    params["train"] = dataset

    # Initialize: Formatter

    dataset = PLDM(params)

    # Initialize: Model

    model = MLP(params)

    # Initialize: Logger 

    if(logger_choice == 0):
        logger = pl_loggers.TensorBoardLogger(path_results, name = "", version = 0)
    else:
        logger = Logger(path_results, name = "", version = 0)

    # Train: Model

    if(use_gpu):

        # Initialize: GPU Trainer

        trainer = Trainer( logger = logger,
                           deterministic = True, 
                           default_root_dir = path_results, 
                           check_val_every_n_epoch = valid_rate,
                           max_epochs = num_epochs, num_nodes = 1, 
                           num_sanity_val_steps = 0, gpus = gpu_list,
                           plugins = DDPPlugin(find_unused_parameters=False, ) )
    else:

        # Initialize: CPU Trainer

        trainer = Trainer( logger = logger,
                           deterministic = True, 
                           max_epochs = num_epochs, 
                           num_sanity_val_steps = 0,
                           default_root_dir = path_results, 
                           check_val_every_n_epoch = valid_rate )
    
    trainer.fit(model, dataset)

