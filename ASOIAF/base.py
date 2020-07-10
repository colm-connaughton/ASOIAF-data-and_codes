#############################################################
#
# base.py
#
# Utility functions that are used throughout the data processing
# pipeline for the 'Network of Thrones' paper. Including reading
# config files, creating a NetworkX network of the interactions, calculating
# interevent times, and calculating empirical pmfs.  
# 
#############################################################


#### Import dependencies

import numpy as np
import os
import yaml
import networkx as nx


#############################################################

#### Utilitiy Functions

def read_config(argv):
    # read_config: 
    #
    # Reads a given config yaml file, whose filename is passed as
    # an argument (argv). If none specified attempts to read the default config
    # file instead. Returns the configuration of the run extracted from the
    # config yaml if it could be read. 
    
    # Determine the name of the config file to be used
    filename='config_default.yaml'
    if len(argv) == 1:
        print("No config file specified. Assuming config_default.yaml")
    else:
        filename = argv[1]
        print("Using config file ", filename, "\n")

    # Check that the config file exists and is readable
    if not os.access(filename, os.R_OK):
        print("Config file ", filename, " does not exist or is not readable. Exiting.")
        exit()

    # Read the config file
    f = open(filename,'r')
    
    # Extract the config information from the yaml file and return it
    config = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    return config





def build_network(characters, interactions):
    # build_network: 
    #
    # Creates networkX network from a dataframe of characters and 
    # a dataframe of their interactions. The characters form the nodes and 
    # their interactions the edges connecting them. Returns the network's 
    # handle.
    
    # Initalise network
    G=nx.Graph()
    
    # Add the nodes and edges
    G.add_nodes_from(characters['name'].values)
    G.add_edges_from(interactions[['char1','char2']].values)
    
    # Return the network, and output a few key stats
    print("    Graph size : ", [G.number_of_nodes(), G.number_of_edges()])
    return G





def calculate_interevent_times(df, mode):
    # calculate_interevent_times: 
    #
    # Calculates the inter-event times between a set of event times given in a 
    # dataframe (df). Has two modes for different formats of event times,
    # chapters for times stored as integers and, date for times in datetime 
    # format. Returns a dataframe of the interevent times. 
    
    # For chapter mode
    if mode == 'chapter':
        # Sort data frame by chapter 
        df=df.sort_values(by=['chapter_index'])
        
        # Calculate IET in chapters by finding difference between event 
        # chapter and the previous event's chapter
        df['prev_chapter'] = df['chapter_index'].shift()
        df=df.dropna()
        df['prev_chapter'] = df['prev_chapter'].astype(int)
        df['interval'] = df['chapter_index'] - df['prev_chapter']
        
        # Drop unused columns from data frame
        df.drop(['prev_chapter'], inplace=True, axis=1)
    
    # For date mode
    elif mode == 'date':
        # Sort data frame by date
        df=df.sort_values(by=['date'])
        
        # Calculate IET as a date by finding difference between date of event 
        # and the previous event's date
        df['prev_date'] = df['date'].shift()
        df=df.dropna()
        df['interval'] = (df['date'] - df['prev_date']).apply(lambda x: x/np.timedelta64(1,'D')).astype(int)
        
        # Drop unused columns from data frame
        df.drop(['prev_date'], inplace=True, axis=1)
    
    # If mode given is not recognised throw an error
    else:
        print("Error: unknown instruction: "+str(mode))

    # Return dataframe of inter-event times
    return df





def calculate_empirical_PMF(data, normalised=False):
    # calculate_empirical_PMF: 
    #
    # Calculates from a list of data its empirical probability mass
    # function and empirical cumulative mass function. 
    
    # Get values appearing in data and their frequency
    value, count = np.unique(data, return_counts=True)
    
    # Calcaultes empirical pmf
    pmf = count / len(data)
    
    # Calculates empirical cmf
    cmf = [0] * len(value)
    cmf[0] = pmf[0]
    for i in range(1,len(value)):
        cmf[i] = cmf[i-1] + pmf[i]
    
    # If normalised = True normalise the values to be on a scale of 0 to 1
    if normalised:
        value=value/np.max(data)
    
    # Return the list of values, pmfs and cmfs
    return value, pmf, cmf
