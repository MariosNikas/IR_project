import textblob as tb

#O indexer είναι υπεύθυνος για την δημιουργία του αντεστραμένου καταλόγου ο οποίος χρησιμοποιείται μελλοντικά στα ερωτήματα αναζήτησης.

def index():
    file = open("Data.csv", "r", encoding='utf8')
    lines = file.readlines()

    # Οι λέξεις αποθηκεύονται με μορφή : {"word" : {"url":7,"url":6 } }
    wordDict = {"a": {"qwe": 2}}
    finalDict = {"a": {"qwe": 2}}
    for line in lines:
        try:
            url, content = line.split("\t")
            for word in content.split():
                try:
                    finalDict[word][url] = finalDict[word][url] + 1
                    wordDict[url][word] = wordDict[url][word] + 1
                except:
                    try:
                        finalDict[str(word)][str(url)] = 1
                    except:
                        finalDict[str(word)] = {}
                        finalDict[str(word)][str(url)] = 1
        except:
            pass

    return finalDict

# Επιστρέφει το λεξικό του κάθε url με τις λέξεις που υπάρχουν σε αυτό.
def count_words():
    wordDict = {}
    try:
        file = open("Data.csv", "r", encoding='utf8')
        lines = file.readlines()
        for line in lines:
            try:
                url, content = line.split("\t")
                words = tb.TextBlob(content).words
                wordDict[url] = words
            except:
                wordDict[url] = words
    except:
        print("Error!")

    return wordDict



