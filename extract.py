#############################################################
#
# extract.py
#
# Script that runs the data cleanup and extraction portion of the
# data-processing pipeline.
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
import ASOIAF.data
import sys

def main(config):
    # main:
    #
    # Run the data extraction scripts determiend by the passed config file



    # Output pipeline progress indicator
    print("Running extract.py.")

    # Count number of chapters in each book
    print("  Counting chapters in each book.")
    cumulative_chapter_count = ASOIAF.data.count_chapters(config)

    # Extract data to allow conversion from discourse to story time
    if config['extract']['steps']['chapters']:
        print("  Building timeline index for all chapters.")
        ASOIAF.data.build_timeline_index(config, cumulative_chapter_count)

    if config['extract']['steps']['POV characters']:
        print("  Extracting POV characters from chapter headings.")
        ASOIAF.data.extract_POV_characters(config)

    # Extract a dataframe of all characters
    if config['extract']['steps']['all characters']:
        print("  Extracting full character list.")
        ASOIAF.data.extract_character_data(config, cumulative_chapter_count)

    # Extract a dataframe of all character interactions
    if config['extract']['steps']['interactions']:
        print("\n\n  Extracting interaction list.")
        ASOIAF.data.extract_interaction_data(config, cumulative_chapter_count)



### If this file is run as a stand-alone script executes the main() function

if __name__ == "__main__":
    # Read the passed config file
    config = ASOIAF.base.read_config(sys.argv)
    # Call main function
    main(config)
