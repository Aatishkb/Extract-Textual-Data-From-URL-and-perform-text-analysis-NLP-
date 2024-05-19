#!/usr/bin/env python
# coding: utf-8

# # Blackcoffer
#  - Test Assignment

# # Extract textual data from URL and perform Sentimental analysis 

# ## Approaches :-
# 
# 1. **Data Extraction:**
#    - Utilize Python programming to extract data from the provided URLs.
#    - Use libraries such as BeautifulSoup, Selenium, or Scrapy for web crawling.
#    - Extract only the article title and text, excluding headers, footers, and other unnecessary content.
#    - Save the extracted articles in text files with URL_ID as their filenames.
# 
# 2. **Data Analysis:**
#    - Employ Python programming for text analysis of the extracted articles.
#    - Compute the variables specified in the "Text Analysis.docx" file, such as positive score, negative score, polarity score, etc.
#    - Refer to the definitions provided in the analysis document for each variable.
#    - Perform computations for each article and save the output in the same order as specified in the "Output Data Structure.xlsx" file.
# 
# 3. **Output Data Structure:**
#    - Ensure the output variables match the format specified in the "Output Data Structure.xlsx" file.
#    - Include all input variables from the "Input.xlsx" file along with the computed variables.
#    - Save the output in CSV or Excel format.
# 
# 4. **Output Variables:**
#   - We have to calculate:
#   - 1.	POSITIVE SCORE
#   - 2.	NEGATIVE SCORE
#   - 3.	POLARITY SCORE
#   - 4.	SUBJECTIVITY SCORE
#   - 5.	AVG SENTENCE LENGTH
#   - 6.	PERCENTAGE OF COMPLEX WORDS
#   - 7.	FOG INDEX
#   - 8.	AVG NUMBER OF WORDS PER SENTENCE
#   - 9.	COMPLEX WORD COUNT
#   - 10.	WORD COUNT
#   - 11.	SYLLABLE PER WORD
#   - 12.	PERSONAL PRONOUNS
#   - 13.	AVG WORD LENGTH
#                   
# 5. **Finaly:** 
#    - After Calculating all the above Parameters Result can be store in .py file

# ## 	Explaining approach solution:

#  - Use Python for both data extraction and analysis.
#  - Utilize libraries such as BeautifulSoup or Scrapy for web crawling.
#  - Extract article text and save it in text files with URL_ID as filenames.
#  - Perform textual analysis to compute specified variables.
#  - Calculate sentiment scores, readability, and other metrics.
#  - Follow defined algorithms for sentiment analysis.
#  - Clean text using stop words lists and create dictionaries for positive and negative words.
#  - Extract derived variables such as Positive Score, Negative Score, Polarity Score, and Subjectivity Score.
#  - Calculate readability metrics like Average Sentence Length, Percentage of Complex Words, and Fog Index.
#  - Count complex words, total words, syllables per word, personal pronouns, and average word length.
#  - Ensure output matches the specified format in the "Output Data Structure.xlsx" file.
#  - Submit solution through provided Google form including .py file, output in CSV or Excel format, and instructions.

# ### (1). Importing Required Libraries :-

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re


# ## Explanation of each library:
# 
# - **pandas (pd)**:
#   - pandas is a powerful data manipulation and analysis library for Python.
#   - It provides data structures like DataFrame and Series, which are widely used for data manipulation and analysis tasks.
#   - pandas allows you to easily read, write, filter, group, and visualize data.
# 
# - **requests**:
#   - requests is a popular HTTP library for making HTTP requests in Python.
#   - It allows you to send HTTP/1.1 requests extremely easily.
# 
# - **BeautifulSoup**:
#   - BeautifulSoup is a Python library for pulling data out of HTML and XML files.
#   - It provides simple methods and Pythonic idioms for navigating, searching, and modifying the parse tree.
#   - BeautifulSoup is commonly used for web scraping tasks.
# 
# - **tqdm**:
#   - tqdm is a fast, extensible progress bar library for Python and CLI.
#   - It provides a simple and easy-to-use progress bar for loops and other iterative processes.
#   - tqdm is particularly useful when working with long-running processes to monitor progress.
# 
# - **nltk**:
#   - Natural Language Toolkit (NLTK) is a leading platform for building Python programs to work with human language data.
#   - It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet.
#   - NLTK also includes a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.
# 
# - **nltk.corpus.stopwords**:
#   - nltk.corpus.stopwords provides a list of common stop words in various languages.
#   - Stop words are words that are filtered out before or after text processing during natural language processing (NLP).
# 
# - **nltk.corpus.opinion_lexicon**:
#   - nltk.corpus.opinion_lexicon provides a list of positive and negative opinion words.
#   - It's commonly used in sentiment analysis tasks to determine the sentiment polarity of text.
# 
# - **collections.defaultdict**:
#   - defaultdict is a subclass of the built-in dict class in Python.
#   - It returns a new dictionary-like object, which is a subclass of the dict class.
#   - defaultdict is useful when you want to have a default value for keys that haven't been set yet in the dictionary.
# 
# - **nltk.tokenize.word_tokenize**:
#   - `word_tokenize` is a function from the NLTK library used for tokenizing text into words.
#   - Tokenization is the process of breaking down a text into smaller units, such as words or sentences, for further analysis.
#   - `word_tokenize` specifically tokenizes text into individual words based on whitespace and punctuation.
# 
# - **nltk.corpus.cmudict**:
#   - `cmudict` is a part of the NLTK library and provides access to the Carnegie Mellon University Pronouncing Dictionary.
#   - This dictionary contains mappings between words and their phonetic representations.
#   - It is commonly used in tasks such as speech processing, text-to-speech synthesis, and syllable counting.
#   - By accessing `cmudict`, you can retrieve the phonetic representation of English words, which is useful for various linguistic analyses and applications.
# 
# - **re**:
#   - Python's built-in regular expression library.
#   - Provides support for working with regular expressions.
#   - Allows for tasks like pattern matching, text manipulation, and parsing.
#   - Useful for tasks such as string cleaning, validation, and extraction.
# 
# - **string**:
#   - Python's built-in module for common string operations.
#   - Contains constants for ASCII letters, digits, and punctuation characters.
#   - Provides functions for string formatting, manipulation, and testing.
#   - Helpful for tasks involving string processing, formatting, and manipulation.

# ### (2). Extract textual data from Single URL

# In[1]:


def extract_article_text(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all paragraphs within the article content
    paragraphs = soup.find_all('p')
    
    # Concatenate text from all paragraphs
    article_text = ' '.join([p.get_text() for p in paragraphs])
    
    return article_text

# Example usage
url = "https://insights.blackcoffer.com/rising-it-cities-and-its-impact-on-the-economy-environment-infrastructure-and-city-life-by-the-year-2040-2/"
article_text = extract_article_text(url)
print(article_text)


# In[4]:


# !pip install openpyxl


# ### (3). Extract textual data from Multiple URL(Excel File)

# In[11]:


def extract_article_text(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all paragraphs within the article content
    paragraphs = soup.find_all('p')
    
    # Concatenate text from all paragraphs
    article_text = ' '.join([p.get_text() for p in paragraphs])
    
    return article_text

def extract_text_from_urls(input_excel, output_excel):
    # Read the input Excel file
    df = pd.read_excel(r'A:\MTECH(Data Science)\NLP\Input.xlsx')
    
    # Initialize a list to store the extracted data
    data = []
    
    # Iterate over each row in the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Text"):
        url_id = row['URL_ID']
        url = row['URL']
        
        # Extract article text from the URL
        article_text = extract_article_text(url)
        
        # Append URL id and article text to the data list
        data.append({'URL_ID': url_id, 'Text': article_text})
    
    # Convert the data list to a dataframe
    result_df = pd.DataFrame(data)
    
    # Save the dataframe to a CSV file
    result_df.to_excel(output_excel, index=False)

# Example usage
input_excel = "input.xlsx"  # Replace with your input Excel file name
output_excel = "Output Data Structure.xlsx"    # Replace with your desired output CSV file name
extract_text_from_urls(input_excel, output_excel)


# ### (4). Load Extracted Textual Data

# In[2]:


df = pd.read_excel(r'Output Data Structure.xlsx')
df.sample(5)


# ### (5). Text Analysis

# #### (i). Cleaning Stop Words

# In[3]:


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

def clean_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Assuming df is your DataFrame containing textual data, and 'Text' is the column name
def clean_stopwords_with_tqdm(df):
    tqdm.pandas(desc="Cleaning Stopwords")
    df['Text'] = df['Text'].progress_apply(clean_stopwords)
    return df

# Example usage
# Replace df with your DataFrame containing the textual data
df = clean_stopwords_with_tqdm(df)


# In[4]:


df.sample(5)


# #### (ii). Creating dictionary of Positive and Negative words

# In[8]:


# Download NLTK opinion_lexicon if not already downloaded
nltk.download('opinion_lexicon')

def create_sentiment_dictionary(df):
    positive_words = defaultdict(int)
    negative_words = defaultdict(int)

    # Iterate over each row in the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Creating Sentiment Dictionary"):
        # Tokenize the text
        tokens = nltk.word_tokenize(row['Text'])  # Assuming 'Text' is the column name containing text
        
        # Classify each token as positive or negative
        for token in tokens:
            if token in opinion_lexicon.positive():
                positive_words[token] += 1
            elif token in opinion_lexicon.negative():
                negative_words[token] += 1

    return positive_words, negative_words

# Example usage
positive_words, negative_words = create_sentiment_dictionary(df)


# In[10]:


positive_words


# In[11]:


negative_words


# ### Extract and Save positive and negative words into Excel

# In[26]:


# Convert defaultdicts to regular dictionaries
positive_words_dict = dict(positive_words)
negative_words_dict = dict(negative_words)

# Create DataFrames from dictionaries
df_positive = pd.DataFrame({'Word': list(positive_words_dict.keys()), 'Frequency': list(positive_words_dict.values())})
df_negative = pd.DataFrame({'Word': list(negative_words_dict.keys()), 'Frequency': list(negative_words_dict.values())})

# Save positive and negative words to Excel
with pd.ExcelWriter('positive_negative_words.xlsx') as writer:
    df_positive.to_excel(writer, sheet_name='Positive', index=False)
    df_negative.to_excel(writer, sheet_name='Negative', index=False)


# In[5]:


df2 = pd.read_excel(r'A:\MTECH(Data Science)\NLP\New folder\positive_negative_words.xlsx')
df2.head(10)


# #### Read the second sheet of the Excel file into a DataFrame

# In[7]:


# Read the second sheet of the Excel file into a DataFrame
df2 = pd.read_excel(r'A:\MTECH(Data Science)\NLP\New folder\positive_negative_words.xlsx', sheet_name=1)

# Display the first 10 rows of the DataFrame
df2.head(10)


# In[12]:


# Read both sheets of the Excel file into separate DataFrames
df1 = pd.read_excel(r'A:\MTECH(Data Science)\NLP\New folder\positive_negative_words.xlsx', sheet_name=0)
df2 = pd.read_excel(r'A:\MTECH(Data Science)\NLP\New folder\positive_negative_words.xlsx', sheet_name=1)

# Display the first 10 rows of each DataFrame
print("First sheet For Positive_Words (df1):")
print(df1.head(10))

print("\nSecond sheet For Nagative_Words (df2):")
print(df2.head(10))


# # **********************************************************************

# ## We have to calculate:
# - 1.	POSITIVE SCORE
# - 2.	NEGATIVE SCORE
# - 3.	POLARITY SCORE
# - 4.	SUBJECTIVITY SCORE
# - 5.	AVG SENTENCE LENGTH
# - 6.	PERCENTAGE OF COMPLEX WORDS
# - 7.	FOG INDEX
# - 8.	AVG NUMBER OF WORDS PER SENTENCE
# - 9.	COMPLEX WORD COUNT
# - 10.	WORD COUNT
# - 11.	SYLLABLE PER WORD
# - 12.	PERSONAL PRONOUNS
# - 13.	AVG WORD LENGTH
# 

# # *********************************************************************

# #### 1. Positive Score:

# In[3]:


# Calculate the positive score
positive_score = df1['Frequency'].sum()

print("Positive Score:", positive_score)


# In[18]:


# Initialize positive score
positive_score = 0

# Calculate positive score
for index, row in df1.iterrows():
    word = row['Word']
    if word in positive_words:
        positive_score += row['Frequency']  # Assigning a value of +1 for each positive word found

# Print the positive score
print("Positive Score:", positive_score)


# In[17]:


# Initialize positive score
positive_score = 0

# Calculate positive score
for index, row in df1.iterrows():
    word = row['Word']
    if word in positive_words:
        positive_score += row['Frequency']  # Assigning a value of +1 for each positive word found

# Print the positive score
print("Positive Score:", positive_score)


# ### 2. Positive Score:

# In[18]:


# Initialize negative score
negative_score = 0

# Calculate negative score
for index, row in df2.iterrows():
    word = row['Word']
    if word in negative_words:
        negative_score -= row['Frequency']  # Assigning a value of -1 for each negative word found

# Print the negative score
print("Negative Score:", -1 * negative_score)


# ### 3. Polarity Score

# In[19]:


# Calculate Polarity Score
polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

# Ensure Polarity Score is within range of -1 to +1
polarity_score = max(-1, min(1, polarity_score))

# Print the Polarity Score
print("Polarity Score:", polarity_score)


# ### Function to calculate total number of words

# In[20]:


# Function to calculate total number of words
def calculate_total_words(text):
    tokens = word_tokenize(text)
    return len(tokens)

# Apply the function to calculate total words for each row and then sum them up
total_words = df['Text'].apply(lambda x: calculate_total_words(x)).sum()

# Print total number of words
print("Total Words in DataFrame:", total_words)


# ### 4. Subjectivity Score

# In[21]:


# Calculate Subjectivity Score
subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)

# Ensure Subjectivity Score is within range of 0 to +1
subjectivity_score = max(0, min(1, subjectivity_score))

# Print the Subjectivity Score
print("Subjectivity Score:", subjectivity_score)


# ### Function to calculate total number of sentences

# In[22]:


# Function to calculate total number of sentences
def calculate_total_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

# Apply the function to calculate total sentences for each row and then sum them up
total_sentences = df['Text'].apply(lambda x: calculate_total_sentences(x)).sum()

# Print total number of sentences
print("Total Sentences in DataFrame:", total_sentences)


# ### 5. Average Sentence Length

# In[23]:


Average_Sentence_Length = total_words / total_sentences
Average_Sentence_Length


# ### Function To Calculate the total number of complex words

# In[25]:


# Download the CMU Pronouncing Dictionary if you haven't already
nltk.download('cmudict')
# Function to calculate the number of complex words in a text
def count_complex_words(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text.lower())  # Convert to lowercase to match CMU dictionary
    complex_word_count = 0
    d = cmudict.dict()
    
    # Check each word
    for word in words:
        # Check if the word is in the CMU dictionary
        if word in d:
            # Check if the word has more than one syllable
            if len([ph for ph in d[word] if any(char.isalpha() for char in ph)]) > 2:
                complex_word_count += 1
    
    return complex_word_count

# Apply the function to each row of the DataFrame with tqdm
tqdm.pandas()
d1 = df['Text'].progress_apply(count_complex_words)

# Calculate the total number of complex words
total_complex_words = d1.sum()

print("Total number of complex words:", total_complex_words)


# ### 6. Percentage of Complex words

# In[26]:


Percentage_of_Complex_words = (total_complex_words / total_words)*100
Percentage_of_Complex_words


# ### 7. Fog Index

# In[27]:


Fog_Index = 0.4 * (Average_Sentence_Length + Percentage_of_Complex_words)
Fog_Index


# ### 8. AVG NUMBER OF WORDS PER SENTENCE

# In[37]:


def average_words_per_sentence(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    total_words = 0

    # Iterate through each sentence to count words
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        total_words += len(words)

    # Calculate average number of words per sentence
    if len(sentences) == 0:
        return 0
    else:
        return total_words / len(sentences)

# Apply the function to each row in the dataframe
d = df['Text'].apply(average_words_per_sentence)

print(d)


# ### 9. COMPLEX WORD COUNT

# In[19]:


# Download CMU Pronouncing Dictionary if not already downloaded
nltk.download('cmudict')

# Load CMU Pronouncing Dictionary and create a set of words
cmu_dict = cmudict.dict()
cmu_words_set = set(cmu_dict.keys())

def count_complex_words(text):
    """
    Function to count the number of complex words in a given text.
    """
    # Tokenize the text using regular expressions
    words = re.findall(r'\b\w+\b', text.lower())
    
    complex_word_count = 0
    for word in words:
        # Check if word is in the set of CMU words
        if word in cmu_words_set:
            # Check if any phoneme contains a digit (indicating multiple syllables)
            if any(any(char.isdigit() for char in phoneme) for phoneme in cmu_dict[word]):
                complex_word_count += 1
    
    return complex_word_count

# Apply the count_complex_words function to each row in the DataFrame with tqdm
tqdm.pandas()
l = df['Text'].progress_apply(count_complex_words)

print(l)


# ### 10. WORD COUNT
# 
# 
# 

# In[5]:


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Function to count cleaned words in the text
def count_cleaned_words(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Remove punctuation marks
    words = [word for word in words if word not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    
    # Count the remaining words
    cleaned_word_count = len(words)
    
    return cleaned_word_count

# Apply the function to each row in the dataframe
c = df['Text'].apply(count_cleaned_words)

print(c)


# ### 11. SYLLABLE PER WORD

# In[12]:


# Your count_syllables function remains the same as before

# Modify count_syllables_in_text to handle words shorter than 4 characters
def count_syllables_in_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Initialize a list to store syllable counts
    syllable_counts = []
    
    # Loop through each word
    for word in words:
        # Count syllables in the word
        syllable_count = count_syllables(word)
        
        # Append the syllable count to the list
        syllable_counts.append(syllable_count)
    
    return syllable_counts



# Apply the function to each row in the dataframe
e = df['Text'].apply(count_syllables_in_text)

print(e)


# In[13]:


# Function to count syllables in a word while handling exceptions
def count_syllables(word):
    # Remove punctuation marks from the word
    word = re.sub(r'[^\w\s]', '', word)
    
    # Count vowels
    vowels = 'aeiou'
    num_vowels = 0
    prev_char = ''
    
    for char in word.lower():
        if char in vowels and prev_char not in vowels:
            num_vowels += 1
        prev_char = char
    
    # Handling exceptions like words ending with "es" or "ed"
    if word.endswith('es'):
        num_vowels -= 1
    elif word.endswith('ed'):
        if len(word) >= 4 and word[-3] in vowels:
            num_vowels += 1
        elif len(word) >= 5 and word[-4] not in vowels:
            num_vowels -= 1
    
    # Avoid negative syllable count
    return max(num_vowels, 1)

# Function to count syllables in each word of the text
def count_syllables_in_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Count syllables in each word
    syllable_counts = [count_syllables(word) for word in words]
    
    return syllable_counts


# Apply the function to each row in the dataframe
e = df['Text'].apply(count_syllables_in_text)

print(e)


# ### 12. PERSONAL PRONOUNS

# In[14]:


# Function to count personal pronouns in the text
def count_personal_pronouns(text):
    # Define the regex pattern to match personal pronouns
    pattern = r'\b(?:I|we|my|ours|us)\b'
    
    # Compile the regex pattern
    regex = re.compile(pattern, flags=re.IGNORECASE)
    
    # Find all matches of personal pronouns in the text
    matches = regex.findall(text)
    
    # Count the number of matches
    count = len(matches)
    
    return count

# Apply the function to each row in the dataframe
f = df['Text'].apply(count_personal_pronouns)

print(f)


# ### 13. AVG WORD LENGTH

# In[8]:


# Function to calculate average word length
def average_word_length(text):
    # Tokenize the text into words
    words = text.split()
    
    # Calculate total number of characters in each word
    total_char_count = sum(len(word) for word in words)
    
    # Calculate total number of words
    total_word_count = len(words)
    
    # Calculate average word length
    if total_word_count == 0:
        return 0
    else:
        return total_char_count / total_word_count

# Apply the function to each row in the dataframe
g = df['Text'].apply(average_word_length)

print(g)


# # *********************************************************************

# ## Name - Aatish Kumar Baitha
#   - M.Tech(Data Science 2nd Year Student)
# - My Linkedin Profile -
#   - https://www.linkedin.com/in/aatish-kumar-baitha-ba9523191
# - My Blog
#   - https://computersciencedatascience.blogspot.com/
# - My Github Profile
#   - https://github.com/Aatishkb

# # Thank you!

# # *********************************************************************
