#############################################################
#
# data.py
#
# Functions used to process the raw data for the 'Network of Thrones' paper.
#
#############################################################


#### Import dependencies

import pandas as pd
import re
import pickle


#############################################################

#### Utility Functions

def read_timeline_data(config):
    # read_timeline_data:
    #
    # Reads the timeline data file and creates a dataframe that relates books
    # and chapters to the year and day on which they start. Takes a pipeline
    # config to extract folder structure from. Returns a dataframe containing
    # a list of books/chapter and their year and day.



    # Exract folder structure from config
    source_folder = config['source data folder']



    # Read PrivateMajor Reddit timeline excel file
    timeline_filename = source_folder+config['timeline data']
    print("    Reading timeline data from ", timeline_filename)
    xl = pd.ExcelFile(timeline_filename)
    timeline = xl.parse("Timeline",
                    skiprows=[1],
                    header=0,
                    usecols=[0,1,2,3,4,5,6,7],
                    index_col=False,
                    parse_dates=False,
                    names=['year','date','event','chapter name','chapter character','book','chapter','cite'])



    # Drop all rows where the event field is empty
    timeline = timeline.dropna(subset=['event'])

    # Keep only the rows which have both a book and a chapter number
    chaptertimes = timeline[~(pd.isnull(timeline['book']) | (pd.isnull(timeline['chapter'])))]

    # Some chapters span a lot of time and have more than one date associated with them. We don't account for this and
    # just drop the rows which duplicate the same book and chapter number. This means we just associate that chapter with
    # the earliest date attributed to it in the timeline
    chaptertimes = chaptertimes.drop_duplicates(['book','chapter'])

    #Drop the single chapter from The Winds of Winter for now
    chaptertimes = chaptertimes[~(chaptertimes['book']=='TWOW')]

    # Sort by book and chapter number
    chaptertimes = chaptertimes.sort_values(by=['book','chapter'])



    # Return data frame containing the chapters and corresponding dates
    return chaptertimes





#############################################################

#### Data Extraction Scripts

def count_chapters(config):
    # count_chapters:
    #
    # Count the number of chapters in each book to allow the the conversion of
    # chapter and book to a single unified number of chapters starting at
    # 1 at the prologue of AGOT and encompasing the entire series.
    # Takes a pipeline config to extract folder structure from. Returns a
    # dictionary of the offsets required to convert a books chapter number to
    # the overall chapter number, indexed by book abbreviation.

    # Read in from timeline a list of chapters (and their dates)
    chapters = read_timeline_data(config)

    # Count how many chapters per book are showing up in the chapterlist :
    chaptercount={}
    gp = chapters.groupby('book')
    for key, item in gp:
        chaptercount[key] = len(item)
    gp.size()

    # Create and returns a dictionary containing the total number of chapters
    # in the preceding books of the series, the required offset for converting
    # book chapter number to overall chapter number.
    cumulative_chapter_count = {'AGOT':0,
            'ACOK':chaptercount['AGOT'],
            'ASOS':chaptercount['AGOT']+chaptercount['ACOK'],
            'AFFC':chaptercount['AGOT']+chaptercount['ACOK'] + chaptercount['ASOS'],
            'ADWD':chaptercount['AGOT']+chaptercount['ACOK'] + chaptercount['ASOS'] + chaptercount['AFFC'],
            'TWOW':chaptercount['AGOT']+chaptercount['ACOK'] + chaptercount['ASOS'] + chaptercount['AFFC'] + chaptercount['ADWD']}
    return cumulative_chapter_count





def build_timeline_index(config, cumulative_chapter_count):
    # build_timeline_index:
    #
    # Processes the timeline data file to create a dataframe that stores chapter
    # information including the date on which they start. Takes a pipeline
    # config to extract folder sturcture from, and the cumulative_chapter_count
    # to allow conversion from book chapter number to an overall chapter
    # number index. Returns nothing.

    # Output a progress marker
    print("  build_timeline_index()")



    ### Import Data

    # Get year and day that each chapter takes place on
    chaptertimes = read_timeline_data(config)

    # Extract folder structure
    target_folder = config['data folder']



    ### Calculate chapter_index and date

    # Assign a chapter_index, a chapter number spanning the entire series
    # starting with chapter 1 being the prologue of AGOT
    chaptertimes['chapter_index'] = 0
    for i, row in chaptertimes.iterrows():
        idx = cumulative_chapter_count[row['book']]+ row['chapter'] + 1
        chaptertimes['chapter_index'].at[i] =  idx

    # Sort by the chapter index
    chaptertimes = chaptertimes.sort_values(by=['chapter_index'])

    # Combine the date and year fields into a datetime object so
    # that we can do arithmetic, plot as a function of date etc
    # We need to add an offset of 1700 to the year since the pandas Timestamp
    # object can't handle pre 1900 dates
    offset=1700
    chaptertimes.year = chaptertimes.year.astype(int) + offset

    # Create a string containing the dates
    chaptertimes['datestr']=chaptertimes['date'].apply(lambda x: x.strftime('%d-%m-%Y')).str.slice(0,6)+chaptertimes['year'].astype(str)
    # Now convert the 'datestr' column into a Timestamp object
    chaptertimes['date'] = pd.to_datetime(chaptertimes['datestr'],  format="%d-%m-%Y")
    # Drop some unneeded columns
    chaptertimes.drop(['datestr', 'year','cite'], inplace=True, axis=1)
    # Set the dataframe index to the chapter_index
    chaptertimes = chaptertimes.set_index('chapter_index')



    ### Output

    # store the chapter information dataframe to a pickle
    outputfile = target_folder+'chapters.pkl'
    chaptertimes.to_pickle(outputfile)






def extract_character_data(config, cumulative_chapter_count):
    # extract_character_data:
    #
    # Processes the book data files to create a dataframe that stores character
    # information such as name, debut, death, and number of chapters alive and
    # dead. Takes a pipeline config to extract folder sturcture from, and the
    # cumulative_chapter_count to allow conversion from book chapter number to
    # an overall chapter number index. Returns nothing.

    # Output a progress marker
    print("  extract_character_data()")



    ### Find Files

    # Extract folder structure
    source_folder = config['source data folder']
    target_folder = config['data folder']
    filenames = config['book data']
    books = config['extract']['books']



    ### Extract Data

    # Set to keep track of which characters have been found already
    characters_found=set()

    # Create dataframe to hold information about characters
    characters = pd.DataFrame(columns=['name','debut', 'death',
                                       'chapters_alive', 'chapters_dead',])

    # Iterate through the book files
    for book in books:
        # Progress Marker
        data_file = source_folder+filenames[book]
        print("\n    Processing ", data_file)

        # Read in the data
        xl = pd.ExcelFile(data_file)
        df = xl.parse()

        # Remove leading and trailing spaces from the 'Character' column
        df['Character'] = df['Character'].str.strip()

        # Figure out which column names need to be checked for interactions
        # There are friendly and hostile interactions listed but the number
        # of such columns is different for each file. So we process header
        # values to find which headers represent interactions of each kind
        column_names = df.columns.values
        friendly_interaction=[]
        hostile_interaction = []
        for item in column_names:
            if 'Friendly' in item:
                friendly_interaction.append(item)
            if 'Hostile' in item:
                hostile_interaction.append(item)



        ### Now process each row of the data files

        c = 1 # Chapter counter for each book
        for i, row in df.iterrows():

            # Define a regular expression to match the chapter separator rows
            m = re.match(r'(CHAPTER *([0-9]*))|(PROLOGUE)|(EPILOGUE)', row['Character'])
            if m:
                # This is a separator row.
                # Work out the chapter number and chapter index
                if (m.group(0)=='PROLOGUE'):
                    c=1
                elif (m.group(0)=='EPILOGUE'):
                    c = c + 1
                else:
                    c = c + 1
                idx = cumulative_chapter_count[book]+ c

                # Output progress marker
                print("\r      ",book+' Chapter = ', str(c)+' Cumulative chapter = '+str(idx), end='')

            else:
                # This is a data row. We process it to extract the character information
                char1 = row['Character']

                # Check if we have found this character already
                if char1 in characters_found:
                    new=False
                else:
                    new=True
                    characters_found.add(char1)

                # Check is this occurence of the character is flagged as a debut
                debut = not pd.isnull(row['Debut'])
                if debut:
                    debut_chapter=idx

                # Check if this occurence of the character is flagged as a death
                if 'Page Out' in column_names:
                    death = not pd.isnull(row['Page Out'])
                if 'Demise' in column_names:
                    death = not pd.isnull(row['Demise'])
                if death:
                    death_chapter=idx

                # Error to catch index exceeding known total of chapters
                if idx > 344:
                    print("Found an index greater than the total number of chapters")
                    exit()

                # If this is a new character, add an entry to the characters dataframe
                if new:
                    newentry = {'name':char1,'debut':idx, 'death':1000}
                    characters = characters.append(newentry, ignore_index=True)
                if debut:
                    characters.loc[characters['name']==char1, 'debut'] = debut_chapter
                if death:
                    characters.loc[characters['name']==char1, 'death'] = death_chapter

                # Check if there are any new characters in the interactions
                F = row[friendly_interaction]
                H = row[hostile_interaction]
                interactions = F.append(H).dropna()

                for char2 in interactions:
                    char2=char2.strip()

                    # Check if we have found this character already
                    if char2 in characters_found:
                        new=False
                    else:
                        new=True
                        characters_found.add(char2)
                    if new:
                        newentry = {'name':char2,'debut':idx, 'death':1000}
                        characters = characters.append(newentry, ignore_index=True)

    ### Cleanup Extracted Data

    # There are a few instances of blank character names in the data. Remove these:
    characters = characters[~(characters['name']=='')]

    # Enforce the debut and death fields being integers
    characters.debut = characters.debut.astype(int)
    characters.death = characters.death.astype(int)


    ### Output

    outputfile = target_folder+'characters.pkl'
    characters.to_pickle(outputfile)


def extract_POV_characters(config):
    # extract_POV_characters:
    #
    # Processes the book data files to create a dataframe that stores information 
    # on how many chapters are from each characters perspective. So called POV
    # characters. akes a pipeline config to extract folder sturcture from. 
    # Returns nothing.
    
    # Output a progress marker
    print("  extract_POV_characters()")
    


    ### Find Files

    # Extract folder structure
    source_folder = config['source data folder']
    target_folder = config['data folder']

    filenames = config['book data']
    books = config['extract']['books']



    ### Extract Data

    # Get list of chapter titles
    chapter_headers = []
    for book in books:
        data_file = source_folder+filenames[book]

        # Read in the data
        xl = pd.ExcelFile(data_file)
        df = xl.parse()
        
        # Remove leading and trailing spaces from the 'Character' column
        df['Character'] = df['Character'].str.strip()
        
        # Check each row to see if is a new chapter
        for i, row in df.iterrows():
            # Define a regular expression to match the chapter separator rows in the spreadsheets
            m = re.match(r'(CHAPTER *([0-9]*))|(PROLOGUE)|(EPILOGUE)', row['Character'])
            if m:
                # This is a separator row. Use it to work out the chapter number and chapter index
                if not (m.group(0)=='PROLOGUE') or (m.group(0)=='EPILOGUE'):
                    chapter_headers.append(row['Character']+book)


    # Process the chapter headers to extract the POV characters and count
    # how many times each appears
    POV_first_names = {}
    for item in chapter_headers:
        # Remove CHAPTER and Number
        item2 = re.sub(r'CHAPTER *([0-9]*)', '', item)
        m = re.search('\(.*?\)', item)
        
        # If Anything is Left
        if m:
            # Remove Roman numerals from chapter titles
            item3 = m.group(0)
            item3 = re.sub(r'((EPILOGUE - )|(-|\(|\))|(".*?"))','', item3)
            item3 = item3.lower()
            item3 = re.sub(r'( xv)','',item3)
            item3 = re.sub(r'( xiv)','',item3)
            item3 = re.sub(r'( xiii)','',item3)
            item3 = re.sub(r'( xii)','',item3)
            item3 = re.sub(r'( xi)','',item3)
            item3 = re.sub(r'( x)','',item3)
            item3 = re.sub(r'( ix)','',item3)
            item3 = re.sub(r'( viii)','',item3)
            item3 = re.sub(r'( vii)','',item3)
            item3 = re.sub(r'( vi)','',item3)
            item3 = re.sub(r'( v)','',item3)
            item3 = re.sub(r'( iv)','',item3)
            item3 = re.sub(r'( iii)','',item3)
            item3 = re.sub(r'( ii)','',item3)
            item3 = re.sub(r'( i)','',item3)
            item3=item3.strip()
        
            # Increment number of chapters that character has appeared in
            if item3 in POV_first_names.keys():
                POV_first_names[item3] += 1
            else:
                POV_first_names[item3] = 1
        
        # If m == False then an error has occured in extracting the
        # character name from the chapter title
        else:
            print('Error: Failed to parse Chapter title "'+item+'"')

    # Name of all POV characters, Found online
    names = {'Daenerys Targaryen' : 'daenerys',
    'Eddard Stark' : 'eddard',
    'Catelyn Stark' : 'catelyn',
    'Jon Snow' : 'jon',
    'Arya Stark' : 'arya',
    'Bran Stark' : 'bran',
    'Tyrion Lannister' : 'tyrion',
    'Sansa Stark' : 'sansa',
    'Davos Seaworth' : 'davos',
    'Theon Greyjoy' : 'theon',
    'Jaime Lannister' : 'jaime',
    'Samwell Tarly' : 'samwell',
    'Merrett Frey' : 'merrett frey',
    'Aeron Greyjoy' : 'aeron',
    'Areo Hotah' : 'areo',
    'Cersei Lannister' : 'cersei',
    'Brienne of Tarth' : 'brienne',
    'Asha Greyjoy' : 'asha',
    'Arys Oakheart' : 'arys',
    'Victarion Greyjoy' : 'victarion',
    'Arianne Martell' : 'arianne',
    'Quentyn Martell' : 'quentyn',
    'Jon Connington' : 'jon connington',
    'Melisandre' : 'melisandre',
    'Barristan Selmy' : 'barristan'}

    # Convert names of POV characters to their full names
    POV_characters = {}
    for i, (key, value) in enumerate(names.items()):
        try:
            POV_characters[key] = POV_first_names[value]
        except:
            pass

    # Create a DataFrame of the results
    POV_characters = pd.DataFrame.from_dict(POV_characters, orient='index', columns=['chapters'])

    # Drop Merret Frey - his chapter is an epilogue and we choose to ignore POV
    # characters in the prologues and epilogues. As seems to be conventionally done
    # by others in discussions of POV characters.
    POV_characters.drop('Merrett Frey', axis=0, inplace=True)




    ###  Output character data to a pickle
    
    outputfile = target_folder+'POV_characters.pkl'
    POV_characters.to_pickle(outputfile)


def extract_interaction_data(config, cumulative_chapter_count):
    # extract_interaction_data:
    #
    # Processes the book data files to create a dataframe that stores character
    # interactions information such as the character involved and in which
    # chapter the interaction takes place. Takes a pipeline config to extract
    # folder sturcture from, and the cumulative_chapter_count to allow
    # conversion from book chapter number to an overall chapter number index.
    # Returns nothing.
    #
    # Requires the character list has been generated by extract_character_data
    # first.

    # Output a progress marker
    print("  extract_interaction_data()")



    ### Find Files

    # Extract folder structure
    source_folder = config['source data folder']
    target_folder = config['data folder']
    filenames = config['book data']
    books = config['extract']['books']



    ### Extract Interaction Data

    # Initalize dictionary to store the lists of interactions and counter
    dict_all = {}
    characters_by_chapter = {}
    counter_all=1

    # Iterate through the book files
    for book in books:
        # Progress Marker
        data_file = source_folder+filenames[book]
        print("\n    Processing ", data_file)

        # Read in the data
        xl = pd.ExcelFile(data_file)
        df = xl.parse()

        # Figure out which column names need to be checked for interactions
        # There are friendly and hostile interactions listed but the number
        # of such columns is different for each file. So we process header
        # values to find which headers represent interactions of each kind
        column_names = df.columns.values
        friendly_interactions=[]
        hostile_interactions=[]
        for item in column_names:
            if 'Friendly' in item:
                friendly_interactions.append(item)
            if 'Hostile' in item:
                hostile_interactions.append(item)



        ### Now process each row of the data files
        c = 1 # Chapter counter for each book
        for i, row in df.iterrows():

            # Define a regular expression to match the chapter separator rows
            m = re.match(r'(CHAPTER *([0-9]*))|(PROLOGUE)|(EPILOGUE)', row['Character'])
            if m:
                # This is a separator row.
                # Work out the chapter number and global chapter index
                if(m.group(0)=='PROLOGUE'):
                    c=1
                elif (m.group(0)=='EPILOGUE'):
                    c = c + 1
                else:
                    c = c + 1
                idx = cumulative_chapter_count[book]+ c
                characters_by_chapter[idx] = set()

                # Output progress marker
                print("\r    "+book+' '+str(c)+' '+str(idx), end='')

            else:
                # This is a data row. We process it to extract the interaction information

                # Get first character in the interaction
                char1 = row['Character'].strip()
                # Skip on if char1 is blank
                if char1 == '':
                    continue
                characters_by_chapter[idx].add(char1.strip())

                # Add any interactions to the interactions dataframe
                row_interactions = row[friendly_interactions].dropna()
                for char2 in row_interactions:
                    char2 = char2.strip()
                    if char2 == '':
                        continue
                    characters_by_chapter[idx].add(char2)
                    newentry = [char1, char2, int(idx), book, int(c)]
                    dict_all[counter_all] = newentry
                    counter_all += 1

                row_interactions = row[hostile_interactions].dropna()
                for char2 in row_interactions:
                    char2 = char2.strip()
                    if char2 == '':
                        continue
                    characters_by_chapter[idx].add(char2)
                    newentry = [char1, char2, int(idx), book, int(c)]
                    dict_all[counter_all] = newentry
                    counter_all += 1

    # Create data frames containing the interaction lists
    columns=['char1','char2', 'chapter_index', 'book', 'chapter']
    interactions_all = pd.DataFrame.from_dict(dict_all, orient='index',
            columns=columns)



    ### Create and Output Interaction Data Frame

    # We need to order the interactions consistently: the link [A,B] should not
    # be different from the link [B,A]. To account for this, we will order
    # every [char1, char2] pair alphabetically so that each pairing appears
    # with a unique order.

    # First define a helper function to sort a single row of the interctions
    # dataframe
    def sort_row(row):
        s =  sorted([row['char1'], row['char2']])
        row['char1'] = s[0]
        row['char2'] = s[1]
        return row

    # Now apply this helper function to each row:
    interactions_all= interactions_all.apply(sort_row, axis=1)
    # This procedure could introduce duplicate edges. Remove any that are there.
    interactions_all = interactions_all.drop_duplicates()

    # Write interaction data to file
    outputfile = target_folder+'interactions_all.pkl'
    interactions_all.to_pickle(outputfile)

    # Write character list by chapter to file
    outputfile = target_folder+'characters_by_chapter.pkl'
    outfile = open(outputfile,'wb')
    pickle.dump(characters_by_chapter,outfile)
    outfile.close()
