#############################################################
#
# plots.py
#
# Script that runs the plotting portion of the data-processing pipeline.
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
    # Run the plotting scripts determiend by the passed config file



    # Output pipeline progress indicator
    print("\nRunning plots.py.")

    # Plot deaths vs time (Figure 4)
    if config['plots']['steps']['deaths vs time']:
        print("  Plotting deaths vs chapter and deaths vs date")
        ASOIAF.figures_and_tables.deaths_vs_time_plots(config)

     # Plot interevent time probability distributions with fits (Figure 5 + 6)
    if config['plots']['steps']['interevent time pdfs']:
        print("  Plotting interevent time pmfs and ccmfs")
        # Best fit parameters from R script
        geom_fit_params = [0.425656]
        zeta_fit_params = [6.0, 2.023124]
        ASOIAF.figures_and_tables.interevent_time_PDF_plots(config, geom_fit_params, zeta_fit_params)

    # Plot characters per chapter and cumulative characters (Figure 2)
    if config['plots']['steps']['chapter counts']:
        print("  Plotting characters by chapter")
        ASOIAF.figures_and_tables.chapter_counts_plots(config)

    # Plot mean degree and assortativity of the network over the chapters (Figure 3)
    if config['plots']['steps']['chapter network properties']:
        print("  Plotting network properties by chapter")
        ASOIAF.figures_and_tables.chapter_network_properties_plots(config)

    # Draw the network of living characters who appear in more than
    # 40 chapters (Figure 1)
    if config['plots']['steps']['important living character network']:
        print("  Plotting network of most important living characters")
        ASOIAF.figures_and_tables.important_living_character_network_plots(config)

    # Draw the network of all characters who appear in more than
    # 40 chapters (Supplementary Figure)
    if config['plots']['steps']['important all character network']:
        print("  Plotting network of most important characters alive or dead")
        ASOIAF.figures_and_tables.important_all_character_network_plots(config)

    # Plot degree distributions
    if config['plots']['steps']['degree distribution']:
        print("  Plotting network degree distributions")
        ASOIAF.figures_and_tables.degree_plot(config)


### If this file is run as a stand-alone script executes the main() function

if __name__ == "__main__":
    # Read the passed config file
    config = ASOIAF.base.read_config(sys.argv)
    # Call main function
    main(config)
