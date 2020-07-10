#############################################################
#
# figures_and_tables.py
#
# Functions to generate all figures and tables in the 'Network of Thrones'
# paper and the associated supplementary material. Requires data files have
# already been generated using the analysis.py script.
#
#############################################################


#### Import dependencies

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta
from scipy.special import zeta
import pickle

from . import base



#############################################################

#### Utility ploting functions to ensure plots in same housestyle

def plot_timeseries(fig, ax, df, column_names, labels, styles, xlabel=None,
                    ylabel=None, legend_location='best', plotTitle=None,
                    ylim=None):
    # plot_timeseries:
    #
    # Plots on the axis (ax) of figure (fig) the columns (column_names) of
    # the dataframe (df) in the styles (styles). Returns the given figure
    # and axes where the data was plotted (fig, ax).
    #
    #
    # Optionally can also set an xlabel, ylabel and Plot Title for the
    # figure. A Legend is always generated if two or more columns plotted
    # in location legend_location. The y limit of the figure can be
    # manully set via the ylim parameter


    # Plot the columns of the dataframe
    for i, item in enumerate(column_names):
        ax.plot(df[item], styles[i], label=labels[i])

    # If y-limit given manually set the axis limit
    if ylim != None:
        ax.set_ylim(ylim)

    # Add legend if more then 1 line to be plotted
    if len(column_names)>1:
        ax.legend(labels, loc=legend_location, fontsize=12)

    # Applies labels to figures axes if given
    if not xlabel == None:
        ax.set_xlabel(xlabel, fontsize=12)
    if not ylabel == None:
        ax.set_ylabel(ylabel, labelpad=-1, fontsize=12)

    # Add title and axes ticks
    ax.set_title(plotTitle, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # Returns the figure handle and axes
    return fig, ax





#############################################################

#### Plotting of Figures

def degree_plot(config):
    # Extract folder structure from config
    source_folder = config['analysis folder']
    target_folder = config['plots folder']
    # Read in the degree counts
    filename = config['analysis folder']+'degree_distribution_all.pkl'
    [deg_all, cnt_all] = pickle.load(open( filename, "rb" ) )
    filename = config['analysis folder']+'degree_distribution_alive.pkl'
    [deg_alive, cnt_alive] = pickle.load(open( filename, "rb" ) )

    # Create plot
    # Setup figure size and axes
    filename = target_folder+'degree_distribution.pdf'
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1,1,1)
    norm = 1.0
    p1 = ax1.bar(np.array(deg_all), np.array(cnt_all)*norm, color='blue', alpha=0.6)
    p2 = ax1.bar(np.array(deg_alive), np.array(cnt_alive)*norm, color='orange', alpha=0.6)
    ax1.set_yscale('log')
    ax1.legend((p2[0], p1[0]), ('Survivor network', 'Full network'))
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title("Degree distributions")

    # Save figure to 'plots folder'
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.90, bottom = 0.20)
    fig.savefig(filename)

def deaths_vs_time_plots(config):
    # deaths_vs_time_plots:
    #
    # Function to produce and store figure 4 of the 'Network of Thrones' Paper,
    # two barcharts showing the number of deaths per chapter and per day
    # throughout the first five books of A Song of Ice and Fire respecitvley.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    deaths_vs_time_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['analysis folder']
    target_folder = config['plots folder']

    # Read in the required data
    filename = source_folder+'death_intervals_by_chapter.pkl'
    deaths_by_chapter = pd.read_pickle(filename)
    filename = source_folder+'death_intervals_by_date.pkl'
    deaths_by_date = pd.read_pickle(filename)

    # Get deaths ordered by chapter
    df = deaths_by_chapter.sort_values(by='chapter_index')
    chaps = df['chapter_index'].values.tolist()
    bodycount_chap = df['body count'].values.tolist()

    # Get deaths ordered by date
    df = deaths_by_date.sort_values(by='date')
    dates = pd.to_datetime(df.date.values)
    bodycount_date = df['body count'].values.tolist()

    # Dates are shifted as a work-around to deal with the fact that Python
    # can't handle dates before 1900.
    dates2 = [item.to_pydatetime()-relativedelta(years=1700) for item in dates]
    days = [int((item - dates[0])/np.timedelta64(1,'D')) for item in dates]
    labels = [(dates2[0]+relativedelta(days=i)).strftime("%b, %Y") for i in [0,200,400,600, 800, 1000] ]



    ### Plotting

    # Setup figure size and axes
    filename = target_folder+'timeseries_of_deaths.pdf'
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)


    # Plot number of deaths per chapter on ax1
    ax1.bar(chaps, bodycount_chap, width=1, color='blue', edgecolor='blue')
    ax1.set_xlabel('Chapter', fontsize=12)
    ax1.set_ylabel('Deaths', fontsize=12)
    ax1.set_title("(A) Number of deaths by chapter")

    # Add highlighted regions to indicate where each book beings and ends
    palette = ['#F3E9D2', '#C6DABF', '#88D498', '#1A936F', '#114B5F']
    txty = 8
    ax1.axvspan(0, 74, facecolor=palette[0], alpha=0.5)
    ax1.axvspan(74, 144, facecolor=palette[1], alpha=0.5)
    ax1.axvspan(144, 226, facecolor=palette[2], alpha=0.5)
    ax1.axvspan(226, 272, facecolor=palette[3], alpha=0.5)
    ax1.axvspan(272, 344, facecolor=palette[4], alpha=0.5)
    ax1.text(18, txty, r'AGOT', fontsize=8)
    ax1.text(90, txty, r'ACOK', fontsize=8)
    ax1.text(170, txty, r'ASOS', fontsize=8)
    ax1.text(235, txty, r'AFFC', fontsize=8)
    ax1.text(290, txty, r'ADWD', fontsize=8)


    # Plot number of deaths per date on ax2
    ax2.bar(days, bodycount_date,width=1, color='blue', edgecolor='blue')
    ax2.set_xticks([0,200,400,600,800,1000])
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Deaths', fontsize=12)
    ax2.set_title("(B) Number of deaths by date")


    # Save figure to 'plots folder'
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.90, bottom = 0.20)
    fig.savefig(filename)





def interevent_time_PDF_plots(config, geom_fit_params, zeta_fit_params):
    # interevent_time_PDF_plots:
    #
    # Function to produce and store figures 5 and 6 of the 'Network of Thrones'
    # paper.
    #
    # Figure 5 consists of two graphs, one of the probabibility mass
    # function and one of the complementary cumulative mass function of the
    # inter-event times between deaths in discourse time (chapters), along with
    # the best fit geometric distribution to this data. The parameters of the
    # geometric distribution is passed in geom_fit_params.
    #
    # Figure 6 consists of two graphs one of the probabibility mass function
    # and one of the complementary cumulative mass function of the inter-event
    # times between deaths in story time (date), along with
    # the best fit zeta distribution to this data. The parameters of the zeta
    # dustribution are passed in zeta_fit_params.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    interevent_time_PDF_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['analysis folder']
    target_folder = config['plots folder']

    # Read in the required data
    filename = source_folder+'death_intervals_by_chapter.pkl'
    deaths_by_chapter = pd.read_pickle(filename)
    filename = source_folder+'death_intervals_by_date.pkl'
    deaths_by_date = pd.read_pickle(filename)

    # Extract the values of the interevent times from the data frames
    deaths_chapter_interval_values = deaths_by_chapter['interval'].values.tolist()
    deaths_date_interval_values = deaths_by_date['interval'].values.tolist()

    # Use these to calculate empirical PMFs
    cvalue, cpmf, ccmf = base.calculate_empirical_PMF(deaths_chapter_interval_values, normalised=False)
    dvalue, dpmf, dcmf = base.calculate_empirical_PMF(deaths_date_interval_values, normalised=False)

    # Calculate the complementary CMFs
    cccmf = [1.0 - x for x in ccmf]
    dccmf = [1.0 - x for x in dcmf]



    ### Calculate fitted geometric and zeta disributions

    # Calculate PMF and CCMF for Fitted Geometric Distribution
    p=geom_fit_params[0]

    max_IET_chapter = max(deaths_chapter_interval_values)
    IETs_chapter = list(range(1,max_IET_chapter+1))

    geom_pmf = [((1.0-p)**(x-1))*p for x in IETs_chapter]
    geom_ccmf = [1.0- (1.0 - (1-p)**(x)) for x in IETs_chapter]

    # Calculate PMF and CCMF for Fitted Zeta Distribution
    xmin = zeta_fit_params[0]
    alpha = zeta_fit_params[1]

    normalisation=1.0-dcmf[int(xmin)-1]  # This fit is only valid for x>= xmin so need to normalise by empirical P(x>=xmin)

    max_IET_date = max(deaths_date_interval_values)
    IETs_date = list(range(int(xmin),max_IET_date+1))

    zeta_pmf = [(normalisation*x**(-alpha))/zeta(alpha,xmin) for x in IETs_date]

    tmp = [0]*len(IETs_date)
    tmp[0]=zeta_pmf[0]
    for i in range(1,len(tmp)):
            tmp[i] = tmp[i-1] + zeta_pmf[i]
    zeta_ccmf = [normalisation - item for item in tmp]



    ### Plotting Figure 5, the by chapter IETs

    # Setup figure size and axes
    filename = target_folder+'deaths_interevent_times_by_chapter.pdf'
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # On ax1 plot pmf for empirical data and geometric fit
    ax1.set_yscale('log')
    ax1.set_xlabel('Inter-event time (chapter)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)

    ax1.plot(cvalue, cpmf, 'ro', label='Empirical data')
    ax1.plot(IETs_chapter, geom_pmf, 'b-', label='Geometric MLE fit')

    ax1.legend(loc='upper right')
    ax1.set_title("(A) Probability Mass Function")

    # On ax2 plot ccmf for empirical data and geometric fit
    ax2.set_yscale('log')
    ax2.set_xlabel('Inter-event time (chapter)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)

    ax2.plot(cvalue[0:-1], cccmf[0:-1], 'ro', label='Empirical data')
    ax2.plot(IETs_chapter, geom_ccmf, 'b-', label='Geometric MLE fit')

    ax2.legend(loc='upper right')
    ax2.set_title("(B) Complementary Cumulative Mass Function")

    # Save figure to 'plots folder'
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    fig.savefig(filename)



    ### Plotting Figure 6, the by date IETs

    # Setup figure size and axes
    filename = target_folder+'deaths_interevent_times_by_date.pdf'
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # On ax1 plot pmf for empirical data and zeta fit
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Inter-event time (day)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.plot(dvalue, dpmf, 'ro', label='Empirical data')
    ax1.plot(IETs_date, zeta_pmf, 'b-', label='Zeta MLE fit')
    ax1.legend(loc='upper right')
    ax1.set_title("(A) Probability Mass Function")

    # On ax1 plot pmf for empirical data and zeta fit
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Inter-event time (day)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.plot(dvalue[0:-1], dccmf[0:-1], 'ro', label='Empirical data')
    ax2.plot(IETs_date, zeta_ccmf, 'b-', label='Zeta MLE fit')
    ax2.legend(loc='upper right')
    ax2.set_title("(B) Complementary Cumulative Mass Function")

    # Save figure to 'plots folder'
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    fig.savefig(filename)





def chapter_counts_plots(config):
    # chapter_counts_plots:
    #
    # Function to produce and store figure 2 of the 'Network of Thrones' Paper.
    # Figure consists of two plots, one showing the number of characters per
    # chapter and the other the total number of unique characters introduced
    # into the story by a given chapter.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    chapter_counts_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['analysis folder']
    target_folder = config['plots folder']

    # Read in the required data
    filename = source_folder+'characters_by_chapter.pkl'
    characters_by_chapter = pd.read_pickle(filename)
    filename = source_folder+'cumulative_characters_by_chapter.pkl'
    cum_characters_by_chapter = pd.read_pickle(filename)




    ### Plotting

    # Setup figure size and axes
    filename = target_folder+'number_of_characters_by_chapter.pdf'
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # On ax1 plot number of characters appearing in each chapter
    title = '(A) Number of characters appearing per chapter'
    fig, ax1 = plot_timeseries(fig, ax1, characters_by_chapter, [ 'total'],
                              [ 'All characters'],
                              [ 'b-'],
                              xlabel='Chapter', ylabel='Number of characters', legend_location='best',
                              plotTitle=title,
                             ylim=[0, 100])

    # On ax2 plot total number of characters and total number of living
    # characters introduced so far in the story
    title = '(B) Evolution of cumulative number of characters'
    fig, ax2 = plot_timeseries(fig, ax2, cum_characters_by_chapter, [ 'Cumulative debuts',  'alive'],
                              [ 'All characters', 'Living characters'],
                              [ 'b-', 'g-'],
                              xlabel='Chapter', ylabel='Cumulative number of characters', legend_location='best',
                              plotTitle=title,
                             ylim=[0, 2100])



    # Add highlighted regions to indicate where each book beings and ends
    palette = ['#F3E9D2', '#C6DABF', '#88D498', '#1A936F', '#114B5F']
    txty = 2.5
    ax1.axvspan(0, 74, facecolor=palette[0], alpha=0.5)
    ax1.axvspan(74, 144, facecolor=palette[1], alpha=0.5)
    ax1.axvspan(144, 226, facecolor=palette[2], alpha=0.5)
    ax1.axvspan(226, 272, facecolor=palette[3], alpha=0.5)
    ax1.axvspan(272, 344, facecolor=palette[4], alpha=0.5)
    ax1.text(18, txty, r'AGOT', fontsize=8)
    ax1.text(90, txty, r'ACOK', fontsize=8)
    ax1.text(170, txty, r'ASOS', fontsize=8)
    ax1.text(235, txty, r'AFFC', fontsize=8)
    ax1.text(290, txty, r'ADWD', fontsize=8)
    txty = 50
    ax2.axvspan(0, 74, facecolor=palette[0], alpha=0.5)
    ax2.axvspan(74, 144, facecolor=palette[1], alpha=0.5)
    ax2.axvspan(144, 226, facecolor=palette[2], alpha=0.5)
    ax2.axvspan(226, 272, facecolor=palette[3], alpha=0.5)
    ax2.axvspan(272, 344, facecolor=palette[4], alpha=0.5)
    ax2.text(18, txty, r'AGOT', fontsize=8)
    ax2.text(90, txty, r'ACOK', fontsize=8)
    ax2.text(170, txty, r'ASOS', fontsize=8)
    ax2.text(235, txty, r'AFFC', fontsize=8)
    ax2.text(290, txty, r'ADWD', fontsize=8)

    # Save figure to 'plots folder'
    fig.savefig(filename)





def chapter_network_properties_plots(config):
    # chapter_network_properties_plot:
    #
    # Function to produce and store figure 3 of the 'Network of Thrones' Paper.
    # Figure consists of two plots, showing respectivley the mean degree
    # and assortativity at the end of each chapter, for both the network of all
    # characters and network of currently living characters.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    chapter_network_properties_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['analysis folder']
    target_folder = config['plots folder']

    # Read in the required data
    filename = source_folder+'network_properties_by_chapter.pkl'
    df = pd.read_pickle(filename)



    ### Plotting

    # Setup figure size and axes
    filename = target_folder+'network_properties_by_chapter.pdf'
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # On ax1 plot mean degree of all character and living character network
    # at the end of each chapter
    title = '(A) Evolution of average degree by chapter'
    fig, ax1 = plot_timeseries(fig, ax1, df, [ 'degree_all', 'degree_alive'],
                              [ 'Full network', 'Survivor network'],
                              [ 'b-', 'g-'],
                              xlabel='Chapter', ylabel='Average degree', legend_location='best',
                              plotTitle=title,
                             ylim=[0, 20])

    # On ax2 plot assorativity of all character and living character network
    # at the end of each chapter
    title = '(B) Evolution of assortativity by chapter'
    fig, ax2 = plot_timeseries(fig, ax2, df, [ 'assortativity_all', 'assortativity_alive'],
                              [ 'Full network', 'Survivor network'],
                              [ 'b-', 'g-'],
                              xlabel='Chapter', ylabel='Assortativity', legend_location='best',
                              plotTitle=title,
                             ylim=[-0.25, 0.25])
    ax2.axhline(y=0, color='k')

    label_x = 195
    label_y = -0.15
    arrow_x = 195
    arrow_y = -0.05

    arrow_properties = dict(
        facecolor="black", width=0.5,
        headwidth=4, shrink=0.1)

    ax2.annotate(
    "Red Wedding", xy=(arrow_x, arrow_y),
    xytext=(label_x, label_y),
    arrowprops=arrow_properties)

    # Add highlighted regions to indicate where each book beings and ends
    palette = ['#F3E9D2', '#C6DABF', '#88D498', '#1A936F', '#114B5F']
    txty = 1
    ax1.axvspan(0, 74, facecolor=palette[0], alpha=0.5)
    ax1.axvspan(74, 144, facecolor=palette[1], alpha=0.5)
    ax1.axvspan(144, 226, facecolor=palette[2], alpha=0.5)
    ax1.axvspan(226, 272, facecolor=palette[3], alpha=0.5)
    ax1.axvspan(272, 344, facecolor=palette[4], alpha=0.5)
    ax1.text(18, txty, r'AGOT', fontsize=8)
    ax1.text(90, txty, r'ACOK', fontsize=8)
    ax1.text(170, txty, r'ASOS', fontsize=8)
    ax1.text(235, txty, r'AFFC', fontsize=8)
    ax1.text(290, txty, r'ADWD', fontsize=8)
    txty = -0.225
    ax2.axvspan(0, 74, facecolor=palette[0], alpha=0.5)
    ax2.axvspan(74, 144, facecolor=palette[1], alpha=0.5)
    ax2.axvspan(144, 226, facecolor=palette[2], alpha=0.5)
    ax2.axvspan(226, 272, facecolor=palette[3], alpha=0.5)
    ax2.axvspan(272, 344, facecolor=palette[4], alpha=0.5)
    ax2.text(18, txty, r'AGOT', fontsize=8)
    ax2.text(90, txty, r'ACOK', fontsize=8)
    ax2.text(170, txty, r'ASOS', fontsize=8)
    ax2.text(235, txty, r'AFFC', fontsize=8)
    ax2.text(290, txty, r'ADWD', fontsize=8)

    # Save figure to 'plots folder'
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    fig.savefig(filename)





def important_living_character_network_plots(config):
    # important_living_character_network_plots:
    #
    # Function to produce and store figure 1 of the 'Network of Thrones' Paper.
    # A graphical depiction of all characters in the network who both appear
    # in 40 or more chapters and are still alive by the end of ADWD.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    important_living_character_network_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['plots folder']

    # Try to read in the list of characters and list of interactions
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
        filename = source_folder+'interactions_all.pkl'
        interactions_all = pd.read_pickle(filename)
    except:
        print("    important_living_character_network_plots() : Failed to read in character or chapter data.")
        print("    Check config to make sure data has been generated.")
        exit()



    ###  Create NetworkX network from the interactions of all characters

    # Initalize Network
    G=nx.Graph()

    # Generate Network Chapter by Chapter
    g = interactions_all.groupby('chapter_index')
    for chap, v in g.groups.items():
        # Update networks with the interactions that took place in this chapter
        df = interactions_all.loc[v]

        # Recording which characters appear in this chapter
        char_in_chap = set([])
        for i, row in df.iterrows():
            char1=row['char1']
            char2=row['char2']

            #Add nodes to network if necessary, record whether character dies
            if (not G.has_node(char1)):
                G.add_node(char1, numChap = 0)
                dies = True
                death = characters.loc[characters['name']==char1, 'death'].values[0]
                if death > 400:
                    dies = False
                G.nodes[char1]['dies'] = dies
            if (not G.has_node(char2)):
                G.add_node(char2, numChap = 0)
                dies = True
                death = characters.loc[characters['name']==char2, 'death'].values[0]
                if death > 400:
                    dies = False
                G.nodes[char2]['dies'] = dies

            #Add characters to set of appearing characters
            char_in_chap.add(char1)
            char_in_chap.add(char2)

            #If edge already present, increments weight for number of interactions
            if (G.has_edge(char1,char2)) or (G.has_edge(char2,char1)):
                if (G.number_of_edges(char1,char2) > 0):
                    G[char1][char2]['weight'] += 1
                else:
                    G[char2][char1]['weight'] += 1

            #Otherwise create the edge
            else:
                 G.add_edge(char1,char2,weight = 1)

        # Increase numChap for all characters appearing in this chapter
        for char in char_in_chap:
            G.nodes[char]['numChap'] += 1


    #Remove dead, zero degree and characters appearing in less then 40 chapters
    rm = []
    for node in G.nodes():
        if G.nodes[node]['dies']:
            rm.append(node)
    G.remove_nodes_from(rm)
    rm = []
    for node in G.nodes():
        if (G.degree(node) == 0):
            rm.append(node)
    G.remove_nodes_from(rm)
    rm = []
    for node in G.nodes():
        if G.nodes[node]['numChap'] < 40:
            rm.append(node)
    G.remove_nodes_from(rm)



    ### Plotting the Network

    #Set node sizes and colour proportional to number of chapters a characeter
    #appears in
    ns = []
    cols = []
    for node in G.nodes():
        ns.append(G.nodes[node]['numChap']*18)
        cols.append(G.nodes[node]['numChap']/200.0)

    #Set edge colour proportional to edge weight, i.e. number of interactions
    ed_cols = []
    for edge in G.edges():
        ed_cols.append(G.edges[edge]['weight']/20.0)

    #Manually set node positions to ensure reproducability
    n_pos = {
      'Jaime Lannister': [0.2,0.4],        #
      'Sandor Clegane': [0.10,0.66],       #
      'Roose Bolton': [-0.86,0.4],         #
      'Petyr Baelish': [-0.3,-0.45],       #
      'Melisandre': [-0.8,-0.2],           #
      'Barristan Selmy': [-0.14,-0.30],    #
      'Benjen Stark': [-0.4,0.70],         #
      'Theon Greyjoy': [-0.6,-0.4],        #
      'Tyrion Lannister': [-0.2,0.4],      #
      'Rickon Stark': [-0.4,0.30],         #
      'Margaery Tyrell': [0.4,-0.2],       #
      'Ilyn Payne': [0.8,0.6],             #
      'Arya Stark': [-0.6,0.50],           #
      'Jorah Mormont': [0.2,-0.25],        #
      'Meryn Trant': [0.50,-0.50],         #
      'Mance Rayder': [-1.0,-0.30],        #
      'Cersei Lannister': [0.0,0.0],       #
      'Myrcella Baratheon': [0.0,0.30],    #
      'Sansa Stark': [-0.30,0.0],          #
      'Tommen Baratheon': [0.24,0.12],     #
      'Daenerys Targaryen': [0.06,-0.45],  #
      'Jon Snow': [-0.6,0.2],              #
      'Edmure Tully': [-0.8,0.8],          #
      'Osmund Kettleblack': [0.75,-0.4],   #
      'Samwell Tarly': [-0.80,0.10],       #
      'Mace Tyrell': [0.6,-0.10],          #
      'Varys': [-0.4,-0.3],                #
      'Bran Stark': [-0.95,0.65],          #
      'Loras Tyrell': [0.6,0.30],          #
      'Walder Frey': [-0.6,0.8],           #
      'Stannis Baratheon': [-0.6,-0.1],    #
      'Bronn': [-0.15,0.7],                #
      'Old Nan': [-1.1,0.4],               #
      'Ramsay Snow': [-1.0,0.0]            #
    }

    #Relabel nodes for aesthetic appeal with newline characters
    lbl = {}
    for node in G.nodes():
        lbl[node] = node.replace(" ", "\n")

    #Draw Network and Save
    plt.figure()
    nx.draw(G,with_labels = True,node_size = ns,pos = n_pos,cmap = 'Greens',\
            vmin = 0.0,vmax=1.2,node_color = cols,labels = lbl, \
            edge_color = '#b3b3b3',edgecolors='k',width=ed_cols,font_size=10)
    filename = target_folder+'important_living_character_network.pdf'
    plt.savefig(filename)





def important_all_character_network_plots(config):
    # important_all_character_network_plots:
    #
    # Function to produce and store a supplementary figure (S1) to the
    # 'Network of Thrones' Paper. A graphical depiction of all characters in
    # the network who appear in 40 or more chapters.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    important_all_character_network_plots()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['plots folder']

    # Try to read in the list of characters and list of interactions
    try:
        filename = source_folder+'interactions_all.pkl'
        interactions_all = pd.read_pickle(filename)
    except:
        print("    important_living_character_network_plots() : Failed to read in character or chapter data.")
        print("    Check config to make sure data has been generated.")
        exit()



    ###  Create NetworkX network from the interactions of all characters

    # Initalize Network
    G=nx.Graph()

     # Generate Network Chapter by Chapter
    g = interactions_all.groupby('chapter_index')
    for chap, v in g.groups.items():
        # Update networks with the interactions that took place in this chapter
        df = interactions_all.loc[v]

        # Recording which characters appear in this chapter
        char_in_chap = set([])
        for i, row in df.iterrows():
            char1=row['char1']
            char2=row['char2']

            #Add nodes to network if necessary
            if (not G.has_node(char1)):
                G.add_node(char1, numChap = 0)
            if (not G.has_node(char2)):
                G.add_node(char2, numChap = 0)

            #Add characters to set of appearing characters
            char_in_chap.add(char1)
            char_in_chap.add(char2)

            #If edge already present, increments weight for number of interactions
            if (G.has_edge(char1,char2)) or (G.has_edge(char2,char1)):
                if (G.number_of_edges(char1,char2) > 0):
                    G[char1][char2]['weight'] += 1
                else:
                    G[char2][char1]['weight'] += 1
            #Otherwise create the edge
            else:
                 G.add_edge(char1,char2,weight = 1)

        # Increase numChap for all characters appearing in this chapter
        for char in char_in_chap:
            G.nodes[char]['numChap'] += 1

    #Remove zero degree and characters appearing in less then 40 chapters
    rm = []
    for node in G.nodes():
        if (G.degree(node) == 0):
            rm.append(node)
    G.remove_nodes_from(rm)
    rm = []
    for node in G.nodes():
        if G.nodes[node]['numChap'] < 40:
            rm.append(node)
    G.remove_nodes_from(rm)



    ### Plotting the Network

    #Set node sizes and colour proportional to number of chapters a characeter
    #appears in
    ns = []
    cols = []
    for node in G.nodes():
        ns.append(G.nodes[node]['numChap']*12)
        cols.append(G.nodes[node]['numChap']/200.0)

    #Set edge colour proportional to edge weight, i.e. number of interactions
    ed_cols = []
    for edge in G.edges():
        ed_cols.append(G.edges[edge]['weight']/20.0)

    #Manually set node positions to ensure reproducability
    n_pos = {
      'Jaime Lannister': [0.2,0.4],        #
      'Sandor Clegane': [0.10,0.66],       #
      'Roose Bolton': [-1.05,0.2],         #
      'Petyr Baelish': [-0.4,-0.55],       #
      'Melisandre': [-0.8,-0.2],           #
      'Barristan Selmy': [-0.14,-0.50],    #
      'Benjen Stark': [-0.4,0.70],         #
      'Theon Greyjoy': [-0.6,-0.4],        #
      'Tyrion Lannister': [-0.2,0.4],      #
      'Rickon Stark': [-0.4,0.50],         #
      'Margaery Tyrell': [0.4,-0.2],       #
      'Ilyn Payne': [0.34,0.22],           #
      'Arya Stark': [-0.6,0.65],           #
      'Jorah Mormont': [0.2,-0.25],        #
      'Meryn Trant': [0.45,-0.50],         #
      'Mance Rayder': [-1.0,-0.30],        #
      'Cersei Lannister': [-0.2,-0.14],    #
      'Myrcella Baratheon': [0.0,0.30],    #
      'Sansa Stark': [-0.40,0.0],          #
      'Tommen Baratheon': [0.24,0.12],     #
      'Daenerys Targaryen': [0.06,-0.43],  #
      'Jon Snow': [-0.6,0.2],              #
      'Edmure Tully': [-0.8,0.8],          #
      'Osmund Kettleblack': [0.65,-0.45],  #
      'Samwell Tarly': [-0.80,0.10],       #
      'Mace Tyrell': [0.6,-0.10],          #
      'Varys': [-0.4,-0.3],                #
      'Bran Stark': [-0.95,0.55],          #
      'Loras Tyrell': [0.8,0.30],          #
      'Walder Frey': [-0.6,0.8],           #
      'Stannis Baratheon': [-0.6,-0.1],    #
      'Bronn': [0.0,0.5],                  #
      'Old Nan': [-1.1,0.4],               #
      'Ramsay Snow': [-1.0,0.0],           #
      'Jeor Mormont': [0.60,-0.30],        #
      'Robb Stark' : [-0.85,0.30],         #
      'Eddard Stark' : [-0.65,0.45],       #
      'Rodrik Cassel' : [-0.25,0.6],       #
      'Catelyn Stark' : [-0.4,0.30],       #
      'Aerys II Targaryen' : [0.22,-0.52], #
      'Maester Luwin' : [-0.2,0.75],       #
      'Jon Arryn' : [0.65,0.65],           #
      'Robert Baratheon' : [0.0,0.1],      #
      'Hoster Tully' : [-1.05,0.75],       #
      'Maester Pycelle' : [0.55,0.15],     #
      'Lysa Arryn' : [0.4,0.55],           #
      'Viserys Targaryen' : [0.25,-0.38],  #
      'Rhaegar Targaryen' : [0.42,-0.35],  #
      'Tywin Lannister' : [-0.25,0.15],    #
      'Joffrey Baratheon' : [0.05,-0.14],  #
      'Renly Baratheon' : [-0.1,-0.35],    #
      'Gregor Clegane' : [0.27,-0.09]      #
    }

    #Relabel nodes for aesthetic appeal with newline characters
    lbl = {}
    for node in G.nodes():
        lbl[node] = node.replace(" ", "\n")
    lbl['Aerys II Targaryen'] = 'Aerys II\nTargaryen' #Keeping to Two Lines Max

    #Draw Network and Save
    plt.figure()
    nx.draw(G,with_labels = True,node_size = ns,pos = n_pos,cmap = 'Greens',\
            vmin = 0.0,vmax=1.2,node_color = cols,labels = lbl, \
            edge_color = '#b3b3b3',edgecolors='k',width=ed_cols,font_size=8)
    filename = target_folder+'important_all_character_network.pdf'
    plt.savefig(filename)





#############################################################

#### Creation of Tables

def degree_betweenness_table(config):
    # degree_betweenness_table:
    #
    # Function to produce and store latex for table 1 of the
    # 'Network of Thrones' Paper. Table lists the top 10 most important
    # characters in the all character network and living only character network
    # by both degree and betweenness centrality. It also lists any  major POV
    # characters who do not appear in the top 10.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.



    # Output pipeline progress indicator
    print("    degree_betweenness_table()")



    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['tables folder']

    # Try to import character network properties data frame.
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
    except:
        print("    degree_betweenness_table() : Failed to read in character data.")
        print("    Check config to make sure data has been generated.")
        exit()



    ### List of POV Characters, both Major and Minor

    Major_POVs = ['Tyrion Lannister','Jon Snow',
    'Arya Stark','Daenerys Targaryen','Catelyn Stark','Sansa Stark','Bran Stark',
    'Jaime Lannister','Eddard Stark','Theon Greyjoy','Davos Seaworth','Cersei Lannister',
    'Samwell Tarly','Brienne Tarth']
    Minor_POVs = ['Aeron Greyjoy','Asha Greyjoy','Victarion Greyjoy',
    'Areo Hotah','Arianne Martell','Arys Oakheart','Jon Connington','Melisandre',
    'Quentyn Martell','Barristan Selmy']
    All_POVs = Major_POVs + Minor_POVs



    ### Produce Table

    # Initalize file to output Latex too
    filename = target_folder+'degree_betweenness_table.txt'
    file = open(filename,"w")

    # Write table caption and header
    file.write(r"""\begin{table}%[tbhp]
\centering
\caption{\label{tab-rankings}Characters ranked by various network attributes}
\begin{tabular}{lr}
All Characters &   \\
\midrule
Degree & Betweenness Centrality \\
\midrule
""")

    ### All Character Network Sub-Table

    # Get sorted list of characters
    characters_by_degree = characters.sort_values('degree_all',ascending = False)
    characters_by_betweenness = characters.sort_values('betweenness_all',ascending = False)

    # Get entries in degree column in format: rank. name (value)
    by_degree_data = []
    for char_num in range(characters_by_degree.shape[0]):
        if ((char_num < 10) or (characters_by_degree.iloc[char_num].get('name') in Major_POVs)):
            if (characters_by_degree.iloc[char_num].get('name') in All_POVs):
                by_degree_data.append(str(char_num+1)+'. ' + characters_by_degree.iloc[char_num].get('name') +
                ' (' + str(int(round(characters_by_degree.iloc[char_num].get('degree_all')))) + ') ')
            else: # Entry in bold if not a POV character
                by_degree_data.append("\\bf{"+str(char_num+1)+'. ' + characters_by_degree.iloc[char_num].get('name') +
                ' (' + str(int(round(characters_by_degree.iloc[char_num].get('degree_all')))) + ')} ')

    # Get entries in betweenness column in format: rank. name (value)
    by_between_data = []
    for char_num in range(characters_by_betweenness.shape[0]):
        if ((char_num < 10) or (characters_by_betweenness.iloc[char_num].get('name') in Major_POVs)):
            if (characters_by_betweenness.iloc[char_num].get('name') in All_POVs):
                by_between_data.append(str(char_num+1)+'. ' + characters_by_betweenness.iloc[char_num].get('name') +
                ' (' + "{:.4f}".format(characters_by_betweenness.iloc[char_num].get('betweenness_all')) + ') ')
            else: # Entry in bold if not a POV character
                by_between_data.append("\\bf{"+str(char_num+1)+'. ' + characters_by_betweenness.iloc[char_num].get('name') +
                ' (' + "{:.4f}".format(characters_by_betweenness.iloc[char_num].get('betweenness_all'))  + ')} ')

    # Pad columns so both the same length
    table_height = max(len(by_between_data),len(by_degree_data))
    by_degree_data = by_degree_data + [' ']*(table_height- len(by_degree_data))
    by_between_data = by_between_data + [' ']*(table_height- len(by_between_data))

    # Correct Brienne's name to be 'of Tarth' rather than 'Tarth'
    for row in range(table_height):
        by_degree_data[row] =   by_degree_data[row].replace("Brienne Tarth","Brienne of Tarth")
        by_between_data[row] =  by_between_data[row].replace("Brienne Tarth","Brienne of Tarth")

    # Ouput all character network subtable to the file
    for row in range(table_height):
        file.write(by_degree_data[row]+'& '+by_between_data[row]+'\\\\'+'\n')
        if row == 9:
            file.write('\\midrule \n')

    file.write(r"""\bottomrule
\end{tabular}
""")



    ### Living Character Network Sub-Table

    # Write subtable caption and header
    file.write(r"""\begin{tabular}{lr}
Alive Characters Only &   \\
\midrule
Degree & Betweenness Centrality \\
\midrule
""")

    # Get sorted list of living characters
    alive_characters = characters.query('death > 500')
    characters_by_degree = alive_characters.sort_values('degree_living',ascending = False)
    characters_by_betweenness = alive_characters.sort_values('betweenness_living',ascending = False)

    # Get entries in degree column in format: rank. name (value)
    by_degree_data = []
    for char_num in range(characters_by_degree.shape[0]):
        if ((char_num < 10) or (characters_by_degree.iloc[char_num].get('name') in Major_POVs)):
            if (characters_by_degree.iloc[char_num].get('name') in All_POVs):
                by_degree_data.append(str(char_num+1)+'. ' + characters_by_degree.iloc[char_num].get('name') +
                ' (' + str(int(round(characters_by_degree.iloc[char_num].get('degree_living')))) + ') ')
            else: # Entry in bold if not a POV character
                by_degree_data.append("\\bf{"+str(char_num+1)+'. ' + characters_by_degree.iloc[char_num].get('name') +
                ' (' + str(int(round(characters_by_degree.iloc[char_num].get('degree_living')))) + ')} ')

    # Get entries in betweenness column in format: rank. name (value)
    by_between_data = []
    for char_num in range(characters_by_betweenness.shape[0]):
        if ((char_num < 10) or (characters_by_betweenness.iloc[char_num].get('name') in Major_POVs)):
            if (characters_by_betweenness.iloc[char_num].get('name') in All_POVs):
                by_between_data.append(str(char_num+1)+'. ' + characters_by_betweenness.iloc[char_num].get('name') +
                ' (' + "{:.4f}".format(characters_by_betweenness.iloc[char_num].get('betweenness_living')) + ') ')
            else: # Entry in bold if not a POV character
                by_between_data.append("\\bf{"+str(char_num+1)+'. ' + characters_by_betweenness.iloc[char_num].get('name') +
                ' (' + "{:.4f}".format(characters_by_betweenness.iloc[char_num].get('betweenness_living'))  + ')} ')

    # Pad columns so both the same length
    table_height = max(len(by_between_data),len(by_degree_data))
    by_degree_data = by_degree_data + [' ']*(table_height- len(by_degree_data))
    by_between_data = by_between_data + [' ']*(table_height- len(by_between_data))

    # Correct Brienne's name to be 'of Tarth' rather than 'Tarth'
    for row in range(table_height):
        by_degree_data[row] =   by_degree_data[row].replace("Brienne Tarth","Brienne of Tarth")
        by_between_data[row] =  by_between_data[row].replace("Brienne Tarth","Brienne of Tarth")

    # Ouput living character network subtable to the file
    for row in range(table_height):
        file.write(by_degree_data[row]+'& '+by_between_data[row]+'\\\\'+'\n')
        if row == 9:
            file.write('\\midrule \n')

    file.write(r"""\bottomrule
\end{tabular}
""")

    # Close Table
    file.write(r"""\addtabletext{Characters ranked by degree and betweenness centrality (with values in parentheses).
The three non-POV characters that appear in the top 10 are highlighted in boldface and predominant POV characters who do not appear in the top 10 are also listed. Qualitatively it appears that the 14 predominant POV characters correlate well with the most important characters by both measures.}
\end{table}
""")




    ### Cleanup

    file.close()








def POV_characters_table(config):
    # all_properties_table:
    #
    # Function to produce and store latex for a supplementary table (S1) to the
    # 'Network of Thrones' Paper. Table lists the POV characters ranked by the
    # number of chapters in which they appear
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.




    # Output pipeline progress indicator
    print("    POV_character_table()")


    ### Import data to plot

    # Exract folder structure from config
    source_folder = config['data folder']
    target_folder = config['tables folder']

    # Try to import character network properties data frame.
    try:
        filename = source_folder+'POV_characters.pkl'
        POV_characters = pd.read_pickle(filename)
    except:
        print("    POV_characters_table() : Failed to read in character data.")
        print("    Check config to make sure data has been generated.")
        exit()


    major = POV_characters[POV_characters['chapters']>5].copy()
    minor = POV_characters[POV_characters['chapters']<=5].copy()
    major.sort_values(by='chapters', ascending=False, inplace=True)
    minor.sort_values(by='chapters', ascending=False, inplace=True)
    ### Produce table data

    data = {}
    counter = 0
    for name1, row in major.iterrows():
        chapters1 = str(row['chapters'])
        if counter < len(minor):
            name2 = minor.iloc[counter].name
            chapters2 = str(minor.iloc[counter]['chapters'])
        else:
            name2=''
            chapters2 = ''
        data[counter] = [name1, chapters1, name2, chapters2]
        counter+=1

    # Initalize file to output Latex too
    filename = target_folder+'POV_character_table.txt'
    file = open(filename,"w")

    # Write table caption and header
    file.write(r"""\begin{table}%[tbhp]
\centering
\caption{\label{tab-POV}POV characters ranked by number of chapters}
\begin{tabular}{ll|ll}
Major POV characters & &Minor POV characters &  \\
Name & Chapters & Name & Chapters  \\
\midrule
""")

    # Ouput all character data to the file
    for key in data:
        row = data[key]
        file.write(row[0] +'& '+row[1]
        +'& '+row[2] +'& '+ row[3]+'\\\\'+'\n')


    file.write(r"""\bottomrule
\end{tabular}
""")

    # Close Table
    file.write(r"""\addtabletext{POV characters ranked by number of chapters.}
\end{table}
""")


    ### Cleanup

    file.close()

def properties_double_table(config, properties1, properties2, headings, num):
    # all_properties_table:
    #
    # Function to produce and store latex for a supplementary table (S1) to the
    # 'Network of Thrones' Paper. Table lists the top 15 most important
    # characters in the all character network and living only character network
    # by various measures. It also lists any major POV characters who do not
    # appear in the top 15.
    #
    # Takes a pipeline config to extract folder structure from. Function does
    # not return a value.

    print("    all_properties_table()")
    source_folder = config['data folder']
    target_folder = config['tables folder']


    ### Read in the characters list with network properties
    try:
        filename = source_folder+'character_network_properties.pkl'
        characters = pd.read_pickle(filename)
    except:
        print("    all_properties_table() : Failed to read in character data.")
        print("    Check config to make sure data has been generated.")
        exit()

    ### POV Characters, Majors must be included in table
    Major_POVs = ['Tyrion Lannister','Jon Snow',
    'Arya Stark','Daenerys Targaryen','Catelyn Stark','Sansa Stark','Bran Stark',
    'Jaime Lannister','Eddard Stark','Theon Greyjoy','Davos Seaworth','Cersei Lannister',
    'Samwell Tarly','Brienne Tarth']
    Minor_POVs = ['Aeron Greyjoy','Asha Greyjoy','Victarion Greyjoy',
    'Areo Hotah','Arianne Martell','Arys Oakheart','Jon Connington','Melisandre',
    'Quentyn Martell','Barristan Selmy']
    All_POVs = Major_POVs + Minor_POVs


    ### Produce Table
    filename = target_folder+'centrality_measures_table.txt'
    #Open and Create Table
    file = open(filename,"w")
    file.write(r"""\begin{table}%[tbhp]
\centering
\caption{\label{tab-rankings}Characters ranked by various network attributes}
\begin{tabular}{""")
    for item in range(0,len(properties1)):
        file.write(r"""l""")
    file.write(r"""}
All Characters""")
    for item in range(1,len(properties1)):
        file.write(r"""  &""")
    file.write(r""" \\
\midrule
""")

    # Write the sub-table
    write_subtable(file, characters, All_POVs, Major_POVs, headings, properties1, num)

    file.write(r"""Surviving characters only""")
    for item in range(1,len(properties2)):
        file.write(r"""  &""")
    file.write(r""" \\
\midrule
""")

    # Write the sub-table
    write_subtable(file, characters, All_POVs, Major_POVs, headings, properties2, num)

    ### Cleanup
    file.write(r"""\bottomrule

\end{tabular}
""")





    #Close Table
    file.write(r"""\addtabletext{Characters ranked by various importance measures (with values in parentheses).
Non-POV characters are highlighted in boldface and predominant POV characters who do not appear in the top 15 are also listed. Qualitatively it appears that the 14 predominant POV characters correlate well with the most important characters by all measures.}
\end{table}
""")
    file.close()


def write_subtable(file, characters, All_POVs, Major_POVs, headings, properties, num):
    for i, item in enumerate(headings):
        file.write(' '+item+' ')
        if not i == len(headings)-1:
            file.write('&')
    file.write(r""" \\
\midrule
""")

    # Prepare the data
    characters_by = {}
    data_by = {}
    #All Character Sub-Table
    for item in properties:
        characters_by[item] =  characters.sort_values(item,ascending = False)
        data_by[item] = []
        for char_num in range(characters_by[item].shape[0]):
            character_name = characters_by[item].iloc[char_num].get('name')
            property_value = characters_by[item].iloc[char_num].get(item)
            # Check for nan: skip this iteration if found
            if property_value != property_value:
                continue
            if (item == 'degree_all') or (item == 'degree_living'):
                property_value = str(int(round(property_value)))
            else:
                property_value = "{:.4f}".format(property_value)

            if ((char_num < num) or (character_name in Major_POVs)):
                if (character_name in All_POVs):
                    data_by[item].append(str(char_num+1)+'. ' + characters_by[item].iloc[char_num].get('name') +
                        ' (' + property_value + ') ')
                else:
                    data_by[item].append("\\bf{"+str(char_num+1)+'. ' + characters_by[item].iloc[char_num].get('name') +
                        ' (' + property_value + ')} ')


    # Work out the table height
    table_height = max([len(data_by[key]) for key in data_by])
    # Fill in blank entries where needed
    for key in data_by:
        n = len(data_by[key])
        data_by[key] = data_by[key] + [' ']*(table_height - n)

    #Correct Brienne's Name
    for row in range(table_height):
        for item in properties:
            data_by[item][row] = data_by[item][row].replace("Brienne Tarth","Brienne of Tarth")

    # Write the table
    for row in range(table_height):
        for i, item in enumerate(properties):
            file.write(' '+ data_by[item][row] + ' ')
            if not i == len(properties) - 1:
                file.write('&')
        file.write('\\\\'+'\n')
        if row == num-1:
            file.write('\\midrule \n')
