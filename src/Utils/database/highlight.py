import nltk
from nltk.corpus import wordnet
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
synonyms = []
antonyms = []

for synset in wordnet.synsets("evil"):
    for l in synset.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print("\n")
print(set(synonyms))
#print(set(antonyms))
pos_tagged_text = nltk.pos_tag(synonyms)
print(pos_tagged_text)
print("\n")
word1 = wordnet.synset('evil.a.01')
word2 = wordnet.synset('malefic.a.01')
print(word1.wup_similarity(word2)*100)

#### highlighting on terminal ####
text = "Hello, world!"
# ANSI escape codes for highlighting 
highlight_start = "\033[43;1m"
highlight_end = "\033[0m"
highlighted_text = f"{highlight_start}{text}{highlight_end}"
print(highlighted_text)
#colorama lib
from colorama import init, Back, Style
# Initialize colorama
init()
# Define the text to be highlighted
text = "Hello, world!"
highlight_start = Back.YELLOW + Style.BRIGHT
highlight_end = Style.RESET_ALL
# Combine the colorama styles with the text
highlighted_text = f"{highlight_start}{text}{highlight_end}"
print(highlighted_text)





