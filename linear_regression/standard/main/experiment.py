#--------------------------------
# Initialize: All System Paths
#--------------------------------

import sys

sys.path.append("../network")

#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import yaml
import argparse

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import train_model 

#--------------------------------
# Remove: Irrelevant Warnings
#--------------------------------

import warnings

warnings.filterwarnings("ignore")

#--------------------------------
# Validate: Configuration File
#--------------------------------

def load_config(argument):

    try:

        return  yaml.load(open(argument), Loader = yaml.FullLoader) 

    except:

        if(argument is None):
            print("\nError: No Configuration File Specified\n")
        else:
            print("\nError: Loading Configuration File") 
            print("\nSuggestion: Check File For Errors")
            print("\nNote: Only Supports YAML Files\n")
        exit()        

#--------------------------------
# Main: Load Configuration File
#--------------------------------

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument( "-config", help = "Experiment Configuration File")
    args = args.parse_args()
   
    params = load_config(args.config) 
    
    train_model.run(params)

