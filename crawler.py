import nltk
import text_processing as tp
import requests
import re
import threading
import csv
from bs4 import BeautifulSoup

nltk.download('stopwords')


# Αυτή η συνάρτηση παίρνει το κείμενο απο τις παραγράφους που βρίσκονται σε ενα url.
def get_text(url):
    # Παίρνουμε τις πληροφορίες της σελίδας.
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # Κρατάμε μόνο τις παραγράφους.
    pars = soup.find_all("p")
    text = ''

    # Παίρνουμε το κείμενο απο τις παραγράφους αλλάζοντας τα new Line & tab.
    for i in pars:
        text += i.get_text()

    text = text.replace("\n", " ").replace("\t", " ")
    text = text.rstrip()
    text = text.encode('utf-8')

    return text


# Αυτή η συνάρτηση εκτελεί το get_text για πολλά link και τα αποθηκεύει.
# Αν ένα link έχει περαστεί ήδη το αγνοεί για να μην υπάρχουν επαναλλήψεις.
def thread_get_text(links, old_content, new_content):
    for i in range(len(links)):
        if links[i] not in old_content:
            new_content[links[i]] = get_text(links[i])


# Αυτή η συνάρτηση δέχεται ως παραμέτρους ένα αντικείμενο "soup" (βιβλιοθήκη beautiful soup 4) και έναν αριθμό (όριο)
# Επιστρέφει μια λίστα με εξερχόμενα link απο το αντικείμενο "soup".
# Το μέγεθος της λίστας είναι μικρότερο η ίσο με το όριο που δίνεται.
def get_links_from(soup, limit):
    links = []
    # Παίρνουμε το περιεχόμενο των συνδέσμων μαζί με τις ετικέτες.
    hrefs = [link.get("href") for link in soup.find_all('a', href=True)]

    # Ο κώδικας βασίστηκε πάνω στο παρακάτω:
    # https://stackoverflow.com/questions/59347372/how-extract-all-urls-in-a-website-using-beautifulsoup
    for item in hrefs:
        # Εδώ παίρνουμε μόνο τους συνδέσμους χωρίς άλλα δεδομένα.
        match = re.search("(?P<url>https?://[^\s]+)", item)
        if match is not None:
            if len(links) == limit:  # όριο σελιδών.
                break
            links.append((match.group("url")))

    # Αν δεν έχουμε φτάσει το όριο των συνδέσμων από τα εξερχόμενα link, ξεκινάμε να αναζητούμε νέους συνδέσμους απο
    # το πρώτο url της λίστας, και κάνουμε την ίδια διαδικασία στη νέα σελίδα μέχρι να φτάσουμε το όριο. Αυτό είναι
    # BFS υλοποίηση καθώς εξαντλούμε τα link μίας σελίδας πριν πάμε στην επόμενη.
    if len(links) != limit:
        new_page = requests.get(links[0])
        new_soup = BeautifulSoup(new_page.content, 'html.parser')
        new_links = get_links_from(new_soup, limit - len(links))
        links += new_links

    return links


# Αυτή η συνάρτηση εκτελεί τη λειτουργεία ενός web crawler για ενα συγκεκριμένα αριθμό απο link.
# Ξεκινάει ο crawler από το url που του δίνουμε εμείς.
# Αυτή η συνάρτηση υποστιρίζει multithreading.
def crawler(limit, n_threads, url):
    fieldnames = ['page', 'content']
    old_pages_content = {}

    try:
        # Διαβάζει τα δεδομένα που έχουν ήδη αντληθεί στο Data.csv.
        with open("Data.csv", mode='r') as file:
            reader = csv.DictReader(file, fieldnames=fieldnames)
            line = 0

            for row in reader:
                if line == 0:
                    line += 1
                    continue

                old_pages_content[row['page']] = row['content']
                line += 1
    except IOError:
        pass

    # Διαβάζουμε το αρχικό url.
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = [url]

    # Παίρνουμε τα εξερχόμενα link με την συνάρτηση get_links_from.
    links += get_links_from(soup, limit)
    pages_content = {}

    # Αξιοποιούμε multithreading στην εξαγωγή του κειμένου απο τα url για να γίνεται πιο αποδοτικά.
    # Το κάθε thread will γράφει στο δικό dictionary το περιεχόμενο και έπειτα συγχονεύονται στη μέθοδο crawler
    dict_for_threads = [{} for _ in range(n_threads)]
    threads = list()

    # Για κάθε thread εκτελούμε την get_text για κάθε link
    for i in range(n_threads):
        # Εξετάζουμε την περίπτωση ο αριθμός των link να μην διαιρείται απο τον αριθμό των thread.
        if i == n_threads - 1:
            batch = links[i * (len(links) // n_threads):]
        else:
            batch = links[i * (len(links) // n_threads):(i + 1) * (len(links) // n_threads)]

        x = threading.Thread(target=thread_get_text, args=(batch, old_pages_content, dict_for_threads[i]))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    # Συγχονεύουμε τα αποτελέσματα του κάθε thread σε ένα dictionary.
    for i in range(n_threads):
        pages_content.update(dict_for_threads[i])

    # Γράφουμε στο csv αρχείο όλο το κείμενο που έχουν εξάγει τα threads.
    with open("Data.csv", mode='a') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")

        for key in pages_content:
            clearContent = tp.text_preprocessing(str(pages_content[key]))
            writer.writerow({'page': key, 'content': clearContent})

    return pages_content
