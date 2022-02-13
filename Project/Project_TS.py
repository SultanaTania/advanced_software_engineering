
# coding: utf-8

# In[65]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
import math


# In[66]:


# read text file
def read_text_file(path):
    book = ""
    with open(path,  encoding="utf8") as f:
        #read and remove empty space from the last of the text
        book = f.read().strip()
    return book


# In[67]:


def splitter(word):
    return [char for char in word]


# In[68]:


# remove all the special character from the file
def remove_special_characters(file):
    
    modified_file = file
   
    # special characters
    special_characters = "\"\"\'\'\“\”#$%&()*+,-./:;!<=>?@[\]^_`{|}~"
    special_char = splitter(special_characters)
 
    # remove each special characters from the input file
    for sp_char in special_char:
        modified_file = modified_file.replace(sp_char, " ")
    
    return modified_file


# In[69]:


# remove all the new line from the file
def remove_new_lines(file):
    return file.replace("\n", " ")


# In[70]:


# convert all the characters to lower case
def convert_to_lowercase(file):
    return file.lower()


# In[71]:


# convert the text into list of words
def text_to_word(text):
    words = text.split()
    
    words = [word for word in words if len(word.strip()) > 0]
    return words


# In[72]:


# covert the list of word to a dictionary with unique words and frequency
def uniq_word_with_freq(words):
    dict = {}
    
    for word in words:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


# In[73]:


# stop word are the words which is used most commonly in a language
def stop_word_percentage(word_dict):
    
    # get english stop words
    stop_words = stopwords.words('english')
    
    # get all words in the document
    total_word = sum(word_dict.values())
    
    # stop word occurence
    sw_counts = sum([v for (k,v) in word_dict.items() if k in stop_words])
     
    occurance = str(round((sw_counts / total_word) * 100, 2)) + "%"
    
    return occurance


# In[74]:


def sum_of_square(data):
    return sum([d*d for d in data])


# In[75]:


def tf_idf_calculate(df1, df2):
    
    df1.rename(columns={'frequency': 'tf'}, inplace=True)
    df2.rename(columns={'frequency': 'tf'}, inplace=True)

    # tf and df of doc-1
    tf_df_1 =  df1['tf'].tolist()

    tf_idf1 = []
    for d in tf_df_1:
        # idf = log2(n/df)
        tf_idf1.append(math.log2(1/d) * d)

    df1["tf_idf"] = tf_idf1
    
    tf_2 =  df2['tf'].tolist()
    tf_idf2 = []

    for d in range(len(df2)):
        tf = df2['tf'].iloc[d]

        df_l = df1.tf[df1.unique_words == df2['unique_words'].iloc[d]]

        df = df_l.values[0] if (len(df_l) > 0) else 0

        tf_idf = (tf * math.log2(1/df)) if (df!=0) else 0

        tf_idf2.append(tf_idf)

    df2["tf_idf"] = tf_idf2

    all_sum = 0
    # get the scaler product of both tf_idf
    for d in range(len(df1)):
        tf_idf1 = df1['tf_idf'].iloc[d]

        tf_idf2 = df2.tf_idf[df2.unique_words == df1['unique_words'].iloc[d]]

        mul = tf_idf1 * tf_idf2.values[0] if (len(tf_idf2) > 0) else 0
        all_sum += mul

    # get sum square of doc-1
    sm1 = math.sqrt(sum_of_square(df1.tf_idf.values))
    sm2 = math.sqrt(sum_of_square(df2.tf_idf.values))

    rad = all_sum / (sm1 * sm2)

    theta = str(round( math.degrees(math.acos(rad)), 2)) + "%"

    return theta


# In[76]:


def count(txt):
    return len(txt)
    
def test_word_count():
    # all words of doc 1
    words = text_to_word(convert_to_lowercase(remove_new_lines(remove_special_characters
                                                  (read_text_file('data/test_count.txt')))))

    assert(count(words) == 6)
    
test_word_count()


# In[77]:


if __name__ == "__main__":
    
    try:
        # read and clean doc 1
        file1 = convert_to_lowercase(remove_new_lines(remove_special_characters
                                                      (read_text_file('data/doc1.txt'))))
        # all words of doc 1
        words1 = text_to_word(file1)
        print(f'Total word in document1: {len(words1)}')

        # get freq of doc 1
        dict1 = uniq_word_with_freq(words1)
        print(f'Total uniqe word in document1: {len(dict1)}')


        # second file read
        file2 = convert_to_lowercase(remove_new_lines(remove_special_characters
                                                      (read_text_file('data/doc2.txt'))))

        # all words of doc 2
        words2 = text_to_word(file2)
        print(f'Total word in document2: {len(words2)}')

        # get freq of doc 2
        dict2 = uniq_word_with_freq(words2)
        print(f'Total uniqe word in document2: {len(dict2)}')


        # first document dict to data frame
        df1 = pd.DataFrame({
            "unique_words" : dict1.keys(),
            "frequency" : dict1.values()
        })

        # sort them using their frequency 
        df1 = df1.sort_values(by=['frequency'], ascending=False)

        # top 10 data of document 1
        df1_header = df1.head(10).to_string(index=False)
        print(f'\nTop ten word occured in document-1 are: \n{df1_header}')


        # second document dict to data frame
        df2 = pd.DataFrame({
            "unique_words" : dict2.keys(),
            "frequency" : dict2.values()
        })

        # sort them using their frequency 
        df2 = df2.sort_values(by=['frequency'], ascending=False)

        # top 10 data of document 1
        df2_header = df2.head(10).to_string(index=False)
        print(f'\nTop ten word occured in document-2 are: \n{df2_header}')

        # stop word occurance
        swp1 = stop_word_percentage(dict1)
        swp2 = stop_word_percentage(dict2)

        print(f'\n\nStop word occurance in document-1: {swp1}')
        print(f'Stop word occurance in document-2: {swp2}')

        similarity = tf_idf_calculate(df1, df2)
        print(f'\n\nBoth document has a similarity of {similarity}')
    
    except ValueError:
        print("Oops! That has some error. Try again...")

