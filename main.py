from flask import Flask, request, render_template, jsonify
from crawler import crawler
from indexer import index, count_words
import query_processor as qr
import nltk
import numpy
import num2words
import requests
import bs4
import textblob

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/join', methods=['GET', 'POST'])
def my_form_post():
    text1 = request.form['text1']
    text2 = request.form['text2']

    inv_idx = index()
    idx = count_words()

    similarity = qr.top_k(qr.read_query(text1), inv_idx, idx, int(text2))

    dict = {}
    for i in similarity.keys():
        dict[i] = i

    return jsonify(result=dict)


if __name__ == '__main__':
    # Εκτυπώνουμε μυνήματα για να πάρουμε τα κατάλληλα στοιχεία από τον χρήστη
    url = input("Εισάγετε την πρώτη σελίδα αναζήτησης του crawler:")
    pages = int(input("Εισάγετε πόσους συνδέσμους θα ψάξει το web crawler:"))
    threads = int(input("Εισάγετε επιθυμητό αριθμό threads:"))
    print("\n\nΟ crawler ξεκίνησε. Παρακαλώ περιμένετε...\n\n\n")
    crawler(pages, threads, url)
    print("Ολοκληρώθηκε η διαδικασία του crawler!\nΠατήστε το link για να μεταβείτε στη μηχανή αναζήτησης.")
    app.run(debug=False)
