#############################################################
#
# analysis.py
#
# Script that runs data analsysi portion of the data-processing pipeline.
# 
#############################################################
#
# Note:
# To set which steps of the pipeline to run pass an appropriate config.yaml
# file. Do not edit the pipeline directly. 
# 
#############################################################



### Import dependencies

import ASOIAF.base
import ASOIAF.analysis
import sys

def main(config):
    # main:
    #
    # Run the analysis scripts determiend by the passed config file
    
    # Output pipeline progress indicator
    print("\n\nRunning analysis.py.")

    # Analyse number of chapters a character appeared in
    if config['analysis']['steps']['count character mentions']:
        print("  Counting number of chapters and interactions for each character.")
        ASOIAF.analysis.count_character_mentions(config)

    # Determine node network properties in final network
    if config['analysis']['steps']['network analysis']:
        print("  Calculating network properties.")
        ASOIAF.analysis.calculate_network_properties(config)

    # Calculate death interevent times
    if config['analysis']['steps']['interevent times']:
        print("  Calculating interevent times for deaths.")
        ASOIAF.analysis.extract_interevent_times(config)

    # Calculating number of characters in each chapter and cumulative numbers
    if config['analysis']['steps']['characters time evolution']:
        print("  Calculating chapter by chapter character numbers")
        ASOIAF.analysis.chapter_by_chapter_characters(config)

    # Calculating chapter by chapter global network properties    
    if config['analysis']['steps']['network time evolution']:
        print("  Calculating chapter by chapter network properties")
        ASOIAF.analysis.chapter_by_chapter_network_properties(config)



### If this file is run as a stand-alone script executes the main() function 

if __name__ == "__main__":
    # Read the passed config file
    config = ASOIAF.base.read_config(sys.argv)
    # Call main function
    main(config)
