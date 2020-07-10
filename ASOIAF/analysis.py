#############################################################
#
# analysis.py
#
# Functions to analyse the inter-event times and networks for the '
# Network of Thrones' paper. Requires raw data files have already been
# processed using the extract.py script.
#
#############################################################


#### Import dependencies

import pandas as pd
import numpy as np
import networkx as nx

import pickle
import collections

from . import base



#############################################################

#### Analysis Scripts

def count_character_mentions(config):
    # count_character_mentions:
    #
    # Counts how many chapters each character appears. Stores this information
    # in the pickeled characters data frame. Information intended for later use
    # as a measure of character importance.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("  count_character_mentions()")


    ### Import data to analyse

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['data folder']

    # Attempt to read in the required characters list and interaction list
    filename = source_folder+'characters.pkl'
    try:
        characters =pd.read_pickle(filename)
    except:
        print("    count_character_mentions() : Failed to read in character list from ", filename)
        print("    Check config to make sure the character list has been generated.")
        exit()

    filename = source_folder+'interactions_all.pkl'
    try:
        interactions_all =pd.read_pickle(filename)
    except:
        print("    count_character_mentions() : Failed to read in interactions list from ", filename)
        print("    Check config to make sure the character list has been generated.")
        exit()


    ### Analysis

    # Initalize new column in characters data frame
    characters['appearances'] = 0

    # Create dictionaries to store sets of chapters each characters appears in
    # indexed by characters name
    keys = characters['name'].values.tolist()
    chaps = [set() for i in  range(0,len(keys))]
    chars_chapters = dict(zip(keys, chaps))

    # Find which chapters characters appear in by seeing in which chapters
    # they have a recorded interaction.
    for i, row in interactions_all.iterrows():
        # Get characters taking part in interaction and the chapter it takes place
        char1=row['char1']
        char2=row['char2']
        chap=row['chapter_index']

        # Update the relevant sets containing the character mention information
        chars_chapters[char1].add(chap)
        chars_chapters[char2].add(chap)

    # Calculate and store number of chapters a character appears in
    characters.set_index('name',inplace=True)
    for char in chars_chapters.keys():
        characters.loc[char, 'appearances'] = len(chars_chapters[char])
    characters.reset_index(inplace=True)

    # Write the updated character data frame to a pickle
    outputfile = target_folder+'characters.pkl'
    characters.to_pickle(outputfile)





def calculate_network_properties(config):
    # calculate_network_properties:
    #
    # Calculates node network properties for living and all characters networks
    # at the end of the current books. Stores this information in a pickled
    # data frame. Information intended for later use to compare
    # measures of character and node importance.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("  calculate_network_properties()")


    ### Import data to analyse

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['data folder']

    # Read in the characters list and interaction list
    try:
        filename = source_folder+'characters.pkl'
        characters = pd.read_pickle(filename)
        filename = source_folder+'interactions_all.pkl'
        interactions_all = pd.read_pickle(filename)
    except:
        print("    calculate_network_properties() : Failed to read in character or interactions data.")
        print("    Check config to make sure data has been generated.")
        exit()



    ### Analysis

    # Find dead characters
    dead_characters = characters.query('death < 500')['name'].values

    # Create network of all and only living
    G = base.build_network(characters, interactions_all)
    livG = G.copy()
    livG.remove_nodes_from(dead_characters)

    # Calculate node properties for each network
    networks = [G,livG]
    suffix = ['_all','_living']
    properties = ['degree','betweenness','eigenvector','closeness','page_rank']

    for net,suf in zip(networks,suffix):
        # Progress indicator
        print("    Processing network"+suf)

        # Calculate the network properties
        degrees = net.degree()
        betweenness = nx.algorithms.centrality.betweenness_centrality(net)
        eigenvector = nx.algorithms.centrality.eigenvector_centrality(net)
        closeness = nx.algorithms.centrality.closeness_centrality(net)
        page_rank = nx.algorithms.link_analysis.pagerank_alg.pagerank(net)

        # Store network properties in dictionary for later lookup
        results = [degrees,betweenness,eigenvector,closeness,page_rank]
        dict_properties = dict(zip(properties, results))

        # Create column names where properties will be added to dataframe
        column_names = [item + suf for item in properties]
        dict_columns = dict(zip(properties, column_names))

        # Update the data frame with network properties for each character
        # First create a new column for each property
        for item in column_names:
            characters[item] = 0.0

        # Next iterate over the characters and store all of their network
        # properties in the relevant columns
        for i, row in characters.iterrows():
            char=row['name']
            for prop in properties:
                if char in dict(dict_properties[prop]).keys():
                    characters[dict_columns[prop]].at[i] = dict_properties[prop][char]
                else: # If do not have this property store NaN
                    characters[dict_columns[prop]].at[i] = np.nan


    # Write updated character data to a new pickle
    outputfile = target_folder+'character_network_properties.pkl'
    characters.to_pickle(outputfile)





def extract_interevent_times(config):
    # extract_interevent_times:
    #
    # Calculates the inter-event time between significant deaths in ASOIAF in
    # both chapters and days. Stores this information in both txt
    # (for R script) and pickled format. Information intended for later use to
    # compare discourse and story time portrayal of events.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("  calculate_interevent_times()")


    ### Import data to analyse

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['analysis folder']

    # Read in the characters list and chapter information
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
        filename = source_folder+'chapters.pkl'
        chapters = pd.read_pickle(filename)
    except:
        print("    calculate_interevent_times() : Failed to read in character or chapter data.")
        print("    Check config to make sure data has been generated.")
        exit()



    ### Analysis

    # Select signficiant character deaths. I.e. character who die, interact with
    # at least one other character, and appear more then once so they are not
    # simply battle fodder character introduced to instantly die
    deaths_df = characters[(characters['death']<1000) & (characters['degree_all'] > 0) & (characters['appearances'] > 1)]
    deaths_df = deaths_df[['name','death']]
    deaths_df = deaths_df.rename(columns={'name':'character', 'death':'chapter_index'})

    # Associate a date with each character death via previously calculated chapter
    # to data conversions
    deaths_df['date'] = pd.to_datetime(chapters.loc[deaths_df['chapter_index'].values.tolist()]['date'].values.tolist())


    # Aggregate multiple deaths occuring in the same chapter together
    gp = deaths_df[['chapter_index', 'character']].groupby('chapter_index')
    # Count how many deaths occur in each chapter
    deaths_by_chapter = gp.count()
    # Rename the 'character' field since it now counts the number of deaths in the chapter
    deaths_by_chapter = deaths_by_chapter.rename(columns={'character':'body count'})
    # Add the chapter_index as a field
    deaths_by_chapter['chapter_index'] = list(deaths_by_chapter.index)
    # Rename the old index
    deaths_by_chapter.index.rename('my index', inplace=True)
    # Calculate the interevent times using the utility function defined in base
    deaths_by_chapter_IET = base.calculate_interevent_times(deaths_by_chapter, 'chapter')


    # Aggregate multiple deaths occuring on the same date together
    gp = deaths_df[['date', 'character']].groupby('date')
    # Count how many deaths occur on each date
    deaths_by_date = gp.count()
    # Rename the 'character' field since it now counts the number of deaths on that date
    deaths_by_date = deaths_by_date.rename(columns={'character':'body count'})
    # Add the date as a field
    deaths_by_date['date'] = list(deaths_by_date.index)
    # Rename the old index
    deaths_by_date.index.rename('my index', inplace=True)
    # Calculate the interevent times using the utility function defined in base
    deaths_by_date_IET = base.calculate_interevent_times(deaths_by_date, 'date')



    ### Output

    # Write the death data frames to disk
    filename = target_folder+'death_intervals_by_chapter.pkl'
    deaths_by_chapter_IET.to_pickle(filename)
    filename = target_folder+'death_intervals_by_date.pkl'
    deaths_by_date_IET.to_pickle(filename)

    # Convert the interevent times data frames into lists so that they can
    # be analysed as distributions
    deaths_chapter_interval_values = deaths_by_chapter_IET['interval'].values.tolist()
    deaths_date_interval_values = deaths_by_date_IET['interval'].values.tolist()

    # Save these lists to files for later analysis
    filename = target_folder+'death_intervals_by_chapter.txt'
    fp = open(filename, 'w')
    for item in deaths_chapter_interval_values:
        fp.write("%s\n" % item)
    fp.close()

    filename = target_folder+'death_intervals_by_date.txt'
    fp = open(filename, 'w')
    for item in deaths_date_interval_values:
        fp.write("%s\n" % item)
    fp.close()





def chapter_by_chapter_characters(config):
    # chapter_by_chapter_characters:
    #
    # Calculates how many characters appears in each chapter and have been
    # cumulativley introduced by the end of that chapter. Stores this
    # information in a pickled data frame. Information intended for later use
    # to compare investigate growth of the network.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("  chapter_by_chapter_characters()")


    ### Import data to analyse

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['analysis folder']

    # Read in the characters list and interaction list
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
        filename = source_folder+'chapters.pkl'
        chapters = pd.read_pickle(filename)
        filename = source_folder+'interactions_all.pkl'
        interactions = pd.read_pickle(filename)
    except:
        print("    chapter_by_chapter_characters() : Failed to read in character or chapter data.")
        print("    Check config to make sure data has been generated.")
        exit()

    # Read dict of character list by chapter from file
    try:
        filename = source_folder+'characters_by_chapter.pkl'
        inputfile = open(filename,'rb')
        characters_by_chapter = pickle.load(inputfile)
        inputfile.close()
    except:
        print("    chapter_by_chapter_characters() : Failed to read in interactions list from ", filename)
        print("    Check config to make sure the character list has been generated.")
        exit()


    ### Number of Characters Introduced and Still Alive, Analysis

    # Get list of chapter numbers
    n = len(chapters)
    chapters = [i+1 for i in  range(0,n)]

    # Initate data structures to store calculated (cumulative) number of
    # characters and deaths as saga progresses
    debuts = [0 for i in  range(0,n)]
    deaths = [0 for i in  range(0,n)]
    cumulativeDebuts = [0 for i in  range(0,n)]
    cumulativeDeaths = [0 for i in  range(0,n)]

    # Count how many deaths and debuts occur by chapter
    debutCounts = characters[['name', 'debut']].groupby('debut').count()
    deathCounts = characters[['name', 'death']].groupby('death').count()
    for i in chapters:
        if i in debutCounts.index:
            debuts[i-1] = debutCounts.loc[i,'name']
        if i in deathCounts.index:
            deaths[i-1] = deathCounts.loc[i,'name']

    # Calculate cumulative sum of deaths and debuts
    cumulativeDebuts[0]=debuts[0]
    cumulativeDeaths[0]=deaths[0]
    for i in range(1,n):
        cumulativeDebuts[i] = cumulativeDebuts[i-1] + debuts[i]
        cumulativeDeaths[i] = cumulativeDeaths[i-1] + deaths[i]

    # Store the cumulative number of deaths and debuts as a pickled data frame
    # adding a difference column giving the number of living characters
    data = list(zip(chapters,cumulativeDebuts, cumulativeDeaths))
    df1 = pd.DataFrame(data, columns=['chapter','Cumulative debuts','Cumulative deaths'])
    df1['alive'] = df1['Cumulative debuts'] - df1['Cumulative deaths']
    df1.set_index('chapter', inplace=True)

    filename = target_folder+'cumulative_characters_by_chapter.pkl'
    df1.to_pickle(filename)



    ### Number of Characters appearing in each Chapter, Analysis

    # Inialize data structures to store number of living and dead characters
    # appearing in each chapter.
    alive = [0 for i in  range(0,n)]
    dead = [0 for i in  range(0,n)]

    # Inialize data structures to store sets of the living and dead characters
    # appearing in each chapter
    alive_chars = [set() for i in  range(0,n)]
    characters_alive = dict(zip(chapters, alive_chars))
    dead_chars = [set() for i in  range(0,n)]
    characters_dead =  dict(zip(chapters, dead_chars))

    for chap in characters_by_chapter:
        characters_in_this_chapter = characters_by_chapter[chap]
        for char in characters_in_this_chapter:
            debut = characters.loc[characters['name']==char, 'debut'].values[0]
            death = characters.loc[characters['name']==char, 'death'].values[0]
            isAlive = True
            if debut == death:
                # Character is probably historical so assumed dead
                isAlive = False
            if chap > death:
                isAlive = False
            if isAlive:
                characters_alive[chap].add(char)
            else:
                characters_dead[chap].add(char)


    # Get number of living and dead characters in each chapter
    for i in chapters:
        alive[i-1] = len(characters_alive[i])
        dead[i-1] = len(characters_dead[i])

    # Store as a pickeled data frame the number of living, dead and total
    # characters appearing in each chapter
    data = list(zip(chapters,alive, dead))
    df2 = pd.DataFrame(data, columns=['chapter','alive','dead'])
    df2['total'] = df2['alive'] + df2['dead']
    df2.set_index('chapter', inplace=True)
    filename = target_folder+'characters_by_chapter.pkl'
    df2.to_pickle(filename)





def chapter_by_chapter_network_properties(config):
    # chapter_by_chapter_network_properties:
    #
    # Calculates global network properties at the end of each chapter for the
    # all character and living character networks. Stores this information in
    # a pickled data frame. Information intended for later use to see if
    # social network is real-world like.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("  chapter_by_chapter_network_properties()")


    ### Import data to analyse

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['analysis folder']

    # Read in the characters, chapter and interaction lists
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
        filename = source_folder+'chapters.pkl'
        chapters = pd.read_pickle(filename)
        filename = source_folder+'interactions_all.pkl'
        interactions_all = pd.read_pickle(filename)
    except:
        print("    chapter_by_chapter_network_properties() : Failed to read in character or chapter data.")
        print("    Check config to make sure data has been generated.")
        exit()


    ### Analysis

    # Get list of chapter numbers
    n = len(chapters)
    chapters = [i+1 for i in  range(0,n)]

    # Initate data structures to store calculated global network properties
    degree_all = [0 for i in  range(0,n)]
    degree_alive = [0 for i in  range(0,n)]
    assortativity_all = [0 for i in  range(0,n)]
    assortativity_alive = [0 for i in  range(0,n)]

    # Initalize the living and all character networks
    G=nx.Graph()
    G_alive=nx.Graph()

    # Iterate over the interactions to assemble networks chapter by chapter
    g = interactions_all.groupby('chapter_index')
    for chap, v in g.groups.items():

        # Get interactions that took place in this chapter (chap)
        df = interactions_all.loc[v]

        # Update network with these interactions
        for i, row in df.iterrows():
            # Extract character names and status of alive or dead for interaction
            char1=row['char1']
            char2=row['char2']
            death1 = characters.loc[characters['name']==char1, 'death'].values[0]
            death2 = characters.loc[characters['name']==char2, 'death'].values[0]
            alive1 = False
            alive2 = False
            if death1 > chap:
                alive1 = True
            if death2 > chap:
                alive2 = True



            # Update G network containing all interactions and characters
            if not char1 in G.nodes():
                G.add_node(char1)
            if not char2 in G.nodes():
                G.add_node(char2)
            G.add_edge(char1, char2)



            # Update G_alive network containing only interactions between characters
            # that are alive in this chapter. Add live characters to node list
            # if they are not already there
            if (alive1 == True) and (not char1 in G_alive.nodes()):
                G_alive.add_node(char1)
            if (alive2 == True) and (not char2 in G_alive.nodes()):
                G_alive.add_node(char2)

            # Add interaction to edge list if both characters are alive
            if (alive1 == True) and (alive2 == True):
                G_alive.add_edge(char1, char2)

            # Remove dead characters from node list. NetworkX should
            # automatically remove any associated edges
            if (alive1 == False) and (char1 in G_alive.nodes()):
                G_alive.remove_node(char1)
            if (alive2 == False) and (char2 in G_alive.nodes()):
                G_alive.remove_node(char2)


        # Calculate network global parameters, mean degree and assortativity
        # For all character network
        degrees = [val for (node, val) in G.degree()]
        degree_all[chap-1] = sum(degrees)/len(degrees)
        assortativity_all[chap-1]=nx.degree_assortativity_coefficient(G)

        # For living character network
        degrees = [val for (node, val) in G_alive.degree()]
        degree_alive[chap-1] = sum(degrees)/len(degrees)
        assortativity_alive[chap-1]=nx.degree_assortativity_coefficient(G_alive)


    ### Output

    # Create a data frame with all the global parameter information chapter
    # by chapter
    data = list(zip(chapters,degree_all, degree_alive, assortativity_all, assortativity_alive))
    df = pd.DataFrame(data, columns=['chapter','degree_all', 'degree_alive', 'assortativity_all', 'assortativity_alive'])
    df.set_index('chapter', inplace=True)

    # Store this data frame in a pkl
    filename = target_folder+'network_properties_by_chapter.pkl'
    df.to_pickle(filename)

    # Calculate the degree distributions for the two networks and save them to
    # a file.
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    filename = target_folder+'degree_distribution_all.pkl'
    pickle.dump( [deg,cnt], open( filename, "wb" ) )

    degree_sequence = sorted([d for n, d in G_alive.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    filename = target_folder+'degree_distribution_alive.pkl'
    pickle.dump( [deg,cnt], open( filename, "wb" ) )
