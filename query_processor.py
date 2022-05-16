import numpy as np
import textblob as tb
import text_processing as tp

# Αυτή η συνάρτηση παίρνει την συμβολοσειρά του ερωτήματος και τη μετατρέπει σε TextBlob WordList.
def read_query(query: str):
    query = tp.text_preprocessing(query)
    query = tb.TextBlob(query).words
    return query

# Αυτή η συνάρτηση επιστρέφει την συχνότητα μίας λέξης σε μία συγκεκριμένη σελίδα.
def freq(word: tb.Word, inv_idx: dict, idx: dict, page: str):
    return inv_idx[word][page] / len(idx[page])


# Αυτή η συνάρτηση επιστρέφει την κανονικοποιημένη συχνότητα μίας λέξης σε μία συγκεκριμένη σελίδα.
def n_freq(word: tb.Word, inv_idx: dict, idx: dict, page: str):
    n_frequency = freq(word, inv_idx, idx, page)

    max_freq = 0
    for j in idx[page]:

        temp = freq(j, inv_idx, idx, page)
        if temp > max_freq:
            max_freq = temp

    n_frequency /= max_freq

    return n_frequency


# Αυτή η συνάρτηση επιστρέφει την Inverse Document Frequency ενός όρου.
def idf(word: tb.Word, inv_idx: dict, idx: dict):
    return np.log(len(inv_idx[word]) / len(idx))


# Αυτή η συνάρτηση επιστρέφει την κανονικοποιημένη Inverse Document Frequency ενός όρου.
def n_idf(word: tb.Word, inv_idx: dict, idx: dict):
    return idf(word, inv_idx, idx) / np.log(len(idx))


# Αυτή η συνάρτηση επιστρέφει τo Βάρος ενός όρου που βρίσκεται μέσα στο ερώτημα.
def weight_in_query(word: tb.Word, inv_idx: dict, idx: dict, query: tb.WordList):
    n_f = query.count(word) / len(query)

    max_freq = 0
    for j in query:

        temp = query.count(j) / len(query)
        if temp > max_freq:
            max_freq = temp

    n_f /= max_freq

    n_inv_df = n_idf(word, inv_idx, idx)
    weight = (n_f / 2 + 0.5) * n_inv_df

    return weight


# Αυτή η συνάρτηση επιστρέφει τo Βάρος ενός όρου που βρίσκεται μέσα σε μια σελίδα.
def weight_in_page(word: tb.Word, inv_idx: dict, idx: dict, page: str):
    if page not in inv_idx[word].keys():
        return 0

    nf = n_freq(word, inv_idx, idx, page)
    nidf = n_idf(word, inv_idx, idx)

    weight = nf * nidf

    return weight


# Αυτή η συνάρτηση επιστρέφει την cosine similarity ενός ερωτήματος με όλες τις σελίδες.
def cosine_similarity(query: tb.WordList, inv_idx: dict, idx: dict):
    similarities = {}

    # # Ελέγχει αν οι όροι του ερωτήματος δεν υπάρχουν στον αντεστραμμένο κατάλογο.
    # # Σε αυτή τη περίπτωση επιστρέφει ειδικό λεξικό.
    valid = False
    for word in query:
        if word in inv_idx.keys():
            valid = True

    if not valid:
        return {"No Results": 0}

    for page in idx.keys():

        prod = sum(weight_in_query(word, inv_idx, idx, query) *
                   weight_in_page(word, inv_idx, idx, page) for word in query)
        mag_q = np.sqrt(np.sum(np.square(weight_in_query(word, inv_idx, idx, query)) for word in query))
        mag_p = np.sqrt(np.sum(np.square(weight_in_page(word, inv_idx, idx, page)) for word in query))

        result = prod / (mag_q * mag_p)

        if np.isnan(result):
            result = 0

        similarities[page] = result

    return similarities


# Αυτή η συνάρτηση επιστρέφει τις Κ πιο ταιριαστές σελίδες σύμφωνα με το ερώτημα.
def top_k(query: tb.WordList, inv_idx: dict, idx: dict, k: int):
    similarities = cosine_similarity(query, inv_idx, idx)



    entries = sorted(similarities, key=similarities.get, reverse=True)[:k]

    top = {}
    for i in entries:
        top[i] = similarities[i]

    return top


# Αυτή η συνάρτηση ανανεώνει το ερώτημα με βάση τον αλγόριθμο Rocchio.
def feedback(query: tb.WordList, relative: [str], non_relative: [str], inv_idx: dict, idx: dict):
    """
    Οι παράμετροι της συνάρτησης έχουν ως εξής :
    :param query: Το ερώτημα πριν καμία αλλαγή.
    :param relative: Είναι μια λίστα απο συμβολοσειρές που περιέχει το σύνολο των σχετικών URL.
    :param non_relative: Είναι μια λίστα απο συμβολοσειρές που περιέχει το σύνολο των μη σχετικών URL.
    :param inv_idx: Ο αντεστραμμένος κατάλογος.
    :param idx: Κανονικός κατάλογος.
    :return: επιστρέφει το ανανεωμένο ερώτημα βάση τις σχετικές σελίδες.
    """

    a, b, c = 2, 5, 0.25  # Παράμετροι που χρησιμοποιούνται στον αλγόριθμο.

    # Διατυπώνουμε το ερώτημα ως διάνυσμα.
    query_vec = np.zeros(len(inv_idx))
    for i in query:
        query_vec[list(inv_idx.keys()).index(i)] = 1

    # Διατυπώνουμε τις σελίδες ως διανύσματα.
    pages_vec = {}
    for i in idx:
        pages_vec[i] = np.zeros(len(inv_idx))

        for j in idx[i]:
            pages_vec[i][list(inv_idx.keys()).index(j)] = inv_idx[j][i]

    # Κρατάμε τα διανύσματα των σχετικών σελίδων.
    relative_vecs = []
    for i in idx.keys():
        if i in relative:
            relative_vecs.append(pages_vec[i])

    relative_term = np.mean(relative_vecs, axis=0)

    # Σε ξεχωριστή λίστα κρατάμε τα διανύσματα μη σχετικών σελίδων.
    non_relative_vecs = []
    for i in idx.keys():
        if i in non_relative:
            non_relative_vecs.append(pages_vec[i])

    non_relative_term = np.mean(non_relative_vecs, axis=0)

    # Υπολογισμός του διανύσματος του ανανεωμένου ερωτήματος.
    new_query_vec = a * query_vec + b * relative_term - c * non_relative_term
    new_query_vec = new_query_vec.clip(min=0)

    # Διαμόρφωση του νέου ερωτήματος.
    new_query = tb.WordList
    words_list = list(inv_idx)

    for i in range(len(inv_idx)):

        if new_query_vec[i] != 0:
            new_query += words_list[i]

    return tb.WordList(new_query)


# Αυτή η συνάρτηση έχει τον ρόλο να ανανεώνει τα αποτελέσματα μιας αναζήτησης βάση των σχολίων του χρήστη.
def update_results(query: tb.WordList, relative: [str], non_relative: [str], inv_idx: dict, idx: dict, k: int):
    """
    Οι παράμετροι της συνάρτησης έχουν ως εξής :
    :param query: Το ερώτημα πριν καμία αλλαγή.
    :param relative: Είναι μια λίστα απο συμβολοσειρές που περιέχει το σύνολο των σχετικών URL.
    :param non_relative: Είναι μια λίστα απο συμβολοσειρές που περιέχει το σύνολο των μη σχετικών URL.
    :param inv_idx: Ο αντεστραμμένος κατάλογος.
    :param idx: Κανονικός κατάλογος.
    :param k: Ο αριθμός αντιπροσωπεύει πόσες σελίδες θα εμφανιστούν στον χρήστη όταν κάνει την αναζήτηση.
    :return: Επιστρέφει το ανανεωμένο ερώτημα βάση τις σχετικές σελίδες.
    """
    new_query = feedback(query, relative, non_relative, inv_idx, idx)

    top = top_k(new_query, inv_idx, idx, k)

    return top
