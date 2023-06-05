from flask import Flask, render_template, request, jsonify
import sqlite3
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

df = pd.read_csv("ua_final.csv")
column_values = df['Place'].unique()

final_compile = []

for i in range(42):
    review1 = []
    df_new = df[df["Place"] == column_values[i]]
    for i in df_new["Review"]:
        review1.append(i)


    def sentence_tokenize(arr):
        final = []
        for i in range(len(arr)):
            a = sent_tokenize(arr[i])
            for j in a:
                final.append(j)
        return final


    sentences1 = sentence_tokenize(review1)


    def words_tokenizer(arr):
        word_list = []
        for i in range(len(arr)):
            words = word_tokenize(arr[i].lower())
            for i in words:
                word_list.append(i)
        return word_list


    words1 = words_tokenizer(sentences1)

    stop = stopwords.words("english")
    cleaned_words1 = [w for w in words1 if not w in stop]


    def get_similar_places(place_name, column_values, matrices, top_n=5):
        place_index = np.where(column_values == place_name)[0][0]
        target_matrix = np.asarray(matrices[place_index])

        similarity_scores = []
        for i, matrix in enumerate(matrices):
            if i != place_index:
                similarity = cosine_similarity(target_matrix, np.asarray(matrix))
                similarity_scores.append((i, similarity))

        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        similar_places = []
        visited = set()
        for index, similarity in similarity_scores:
            place = column_values[index]
            if place not in visited:
                similar_places.append(place)
                visited.add(place)
                if len(similar_places) == top_n:
                    break

        return similar_places


    def remove_unwanted_characters(words):
        cleaned_words = []
        pattern = r"[^\w\s]"  # Regex pattern to match non-alphanumeric characters

        for word in words:
            cleaned_word = re.sub(pattern, "", word)
            if cleaned_word:
                cleaned_words.append(cleaned_word)

        return cleaned_words


    final_cleaned1 = remove_unwanted_characters(cleaned_words1)


    def join_words(word_array):
        word_list = ','.join(word_array).split(',')
        return ' '.join(word_list)


    sentence1 = join_words(final_cleaned1)

    final_compile.append(sentence1)

    count_vec = CountVectorizer(max_features=1000)
    x_train_features = count_vec.fit_transform(final_compile)
    a = x_train_features.todense()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/search-results', methods=['GET'])
def search_results():
    query = request.args.get('query')

    conn = sqlite3.connect('test_database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT distinct Place FROM cosine WHERE Review LIKE ?",
                   (f"%{query}%",))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('search_results.html', results=results)


@app.route('/place-description/<place>')
def place_description(place):
    conn = sqlite3.connect('test_database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT about FROM explore WHERE place = ?", (place,))
    description = cursor.fetchone()[0]

    similar_places = get_similar_places(place, column_values, a)

    # Create a list of tuples containing similar place and index
    similar_places_with_index = [(similar_place, index) for index, similar_place in enumerate(similar_places)]

    cursor.close()
    conn.close()

    return render_template('place_description.html', place=place, description=description,
                           similar_places=similar_places)



@app.route('/result', methods=["GET", "POST"])
def result():
    place = request.form.get('place')

    
    conn = sqlite3.connect('test_database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO planning (place) VALUES (?)", (place,))
    conn.commit()
    #conn.close()

    return render_template('result.html', place=place)





if __name__ == '__main__':
    app.run(debug=True)
