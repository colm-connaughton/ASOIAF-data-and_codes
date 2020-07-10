#############################################################
#
# tables.py
#
# Script that runs the tabulating portion of the data-processing pipeline.
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
import ASOIAF.figures_and_tables
import sys

def main(config):
    # main:
    #
    # Run the tabulating scripts determiend by the passed config file



    # Output pipeline progress indicator
    print("\nRunning tables.py.")

    # Tabulate important characters by degree and betweenness (Table 1)
    if config['tables']['steps']['degree and betweenness']:
        print("  Tabulating degree and betweenness of important characters")
        ASOIAF.figures_and_tables.degree_betweenness_table(config)

    # Tabulate important characters by various measures (Supplementary Table)
    if config['tables']['steps']['all properties']:
        print("  Tabulating all network properties for important characters")
        props1 = ['betweenness_all', 'closeness_all', 'page_rank_all','eigenvector_all']
        props2 = ['betweenness_living', 'closeness_living', 'page_rank_living', 'eigenvector_living']
        headings1 = ['Betweenness Centrality', 'Closeness Centrality', 'PageRank', 'Eigenvector Centrality']
        ASOIAF.figures_and_tables.properties_double_table(config, props1, props2, headings1, 15)

    # Tabulate important characters by various measures (Supplementary Table)
    if config['tables']['steps']['POV characters']:
        print("  Tabulating POV characters")
        ASOIAF.figures_and_tables.POV_characters_table(config)



### If this file is run as a stand-alone script executes the main() function

if __name__ == "__main__":
    # Read the passed config file
    config = ASOIAF.base.read_config(sys.argv)
    # Call main function
    main(config)
