import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('wordnet')
# Import packages
import pandas as pd
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
# Instantiate stemmers and lemmatiser
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatiser = WordNetLemmatizer()
# Create function that normalises text using all three techniques
def normalised_text(list, pos='v'):
    """Stem and lemmatise each word in a list. Return output in a dataframe."""
    normalised_text = pd.DataFrame(index=list, columns=['Porter', 'Lancaster', 'Lemmatiser'])
    for word in list:
        normalised_text.loc[word,'Porter'] = porter.stem(word)
        normalised_text.loc[word,'Lancaster'] = lancaster.stem(word)
        normalised_text.loc[word,'Lemmatiser'] = lemmatiser.lemmatize(word, pos=pos)
    return normalised_text
s = input("enter your sentence: ")
count = 0
count = s.split()
if len(count) in range(40):
    l = word_tokenize(s)
    print(normalised_text(l))
else:
    print("enter a shorter sentence")
    s1 = input("enter your sentence: ")
    print(word_tokenize(s1))

#text = s.split()
#print(text)
#print(normalise_text(text))
#normalise_text(['apples', 'pears', 'tasks', 'children', 'earrings', 'dictionary', 'marriage', 'connections', 'universe', 'university'], pos='n')
