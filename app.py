import flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

# Load your dataset
df2 = pd.read_csv('tmdb.csv')

# Create the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', analyzer='word')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['soup'])

# Construct the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset the DataFrame index and create movie indices
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Create an array with all movie titles
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return DataFrame with similar movies
    return_df = pd.DataFrame(columns=['Title', 'Homepage', 'ReleaseDate'])
    return_df['Title'] = df2['title'].iloc[movie_indices]
    return_df['Homepage'] = df2['homepage'].iloc[movie_indices].apply(lambda x: x if len(str(x)) > 3 else "#")
    return_df['ReleaseDate'] = df2['release_date'].iloc[movie_indices]

    return return_df

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        m_name = " ".join(flask.request.form['movie_name'].title().split())
        if m_name not in all_titles:
            return flask.render_template('notFound.html', name=m_name)
        else:
            result_final = get_recommendations(m_name)
            names = result_final['Title'].tolist()
            homepage = result_final['Homepage'].tolist()
            releaseDate = result_final['ReleaseDate'].tolist()

            return flask.render_template('found.html', movie_names=names, movie_homepage=homepage, search_name=m_name, movie_releaseDate=releaseDate)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
