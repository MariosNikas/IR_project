import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from num2words import num2words

# Επεξεργασία κειμένου για την καλύτερη διαχείριση του

# Συνάρτηση που μετατρέπει όλους τους χαρακτήρες απο κεφαλαίους σε μικρούς.
def convert_lower_case(data):
    return np.char.lower(data)

# Συνάρτηση που αφαιρεί ανεπιθύμητες λέξεις ή άρθρα ( πχ. a, the, is, are) που δεν μας ενφιαφέρουν.
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

# Η συνάρτηση αυτή αφαιρεί τα σημεία στίξης.
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

# Η συνάρτηση αυτή αφαιρεί τους απόστροφους.
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

# Η συνάρτηση αυτή μετατρέπει κάθε λέξη στη ρίζα της για καλύτερη διαχείρηση.
def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

# Η συνάρτηση αυτή μετατρέπει οποιοδήποτε αριθμό σε κείμενο.
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


# Αυτή η συνάρτηση καλεί τις προηγούμενες συναρτήσεις του text_processing.py και αφαιρεί τις ασήμαντες λέξεις,
# τα σημεία στίξης, τους αριθμούς και μετατρέπει κάθε λέξη στην ρίζα της για καλύτερη διαχείρηση τους.
def text_preprocessing(text: str):
    clearContent = re.sub("[^0-9a-zA-z ]", " ", text.replace("\\", " "))
    clearContent = convert_lower_case(clearContent)
    clearContent = remove_punctuation(clearContent)
    clearContent = remove_apostrophe(clearContent)
    clearContent = remove_stop_words(clearContent)
    clearContent = convert_numbers(clearContent)
    clearContent = stemming(clearContent)
    clearContent = remove_punctuation(clearContent)
    clearContent = convert_numbers(clearContent)
    clearContent = stemming(clearContent)
    clearContent = remove_punctuation(clearContent)
    clearContent = remove_stop_words(clearContent)

    return clearContent
