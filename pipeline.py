#############################################################
#
# pipeline.py
#
# Script that runs entire data-processing pipeline.
# 
#############################################################
#
# Note:
# To set which steps of the pipeline to run pass an appropriate config.yaml
# file. Do not edit the pipeline directly. 
# 
#############################################################


### Import dependencies

import sys
import ASOIAF.base



def main():
    # main:
    #
    # Runs the portions of the pipeline as determiend by the passed config
    # file, or config_default.yaml if none passed.
    
    
    
    # Read the passed config file
    config = ASOIAF.base.read_config(sys.argv)

    # Extract the data from raw source data folder into a clean format
    if config['extract']['run']:
        import extract
        extract.main(config)

    # Perform the different aspects of the analysis
    if config['analysis']['run']:
        import analysis
        analysis.main(config)

    # Create figures from the analysis results
    if config['plots']['run']:
        import plots
        plots.main(config)

    # Create tables from the analysis results
    if config['tables']['run']:
        import tables
        tables.main(config)


### On being called execute the main() function
        
if __name__ == "__main__":
    main()
