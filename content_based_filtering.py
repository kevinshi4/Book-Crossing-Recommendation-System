import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_colwidth', None)

# Load preprocessed book data
preprocessed_df = pd.read_csv('/Users/chrisfluta/PycharmProjects/DSCI351/Book reviews/Preprocessed_data.csv')

# Drop irrelevant columns
df = preprocessed_df.drop(['Unnamed: 0', 'location', 'img_s', 'img_m', 'img_l'], axis=1)

# Ensure data types are correct
df['age'] = df['age'].astype(int)
df['year_of_publication'] = df['year_of_publication'].astype(int)

# Filter data
df = df[df['country'].notnull()]
df = df[df['country'].str.contains('usa', case=False)]
df = df[df['Language'].str.contains('en', case=False)]

# Further filtering based on user and isbn counts
userCounts = df['user_id'].value_counts()
isbnCounts = df['isbn'].value_counts()
df = df[df['user_id'].isin(userCounts[userCounts >= 5].index)]
df = df[df['isbn'].isin(isbnCounts[isbnCounts >= 20].index)]

####################################
# Select only the relevant columns
####################################
relevant_columns = [
    'book_title',
    'book_author',
    'Summary',
    'Category',
    'publisher',
    'isbn',
    'year_of_publication',
    'Language'
]

# Filter the DataFrame to keep only these columns
df = df[relevant_columns]

# Drop any exact duplicates, then drop duplicate book titles and isbn due to multiple versions
df.drop_duplicates(inplace=True)
df = df.drop_duplicates(subset='book_title', keep='first')
df = df.drop_duplicates(subset='isbn', keep='first')
df = df.drop_duplicates(subset='Summary', keep='first')

# Reset indices to align with the similarity matrix
df = df.reset_index(drop=True)

# Combine author, summary, genre, and publisher into a single string
df['combined_features'] = df.apply(lambda x: f"{x['book_author']} {x['Summary']} {x['Category']} {x['publisher']}",
                                   axis=1)

# Strip combined features
df['combined_features'] = df['combined_features'].str.replace(r"[^\w\s]", "", regex=True).str.lower()

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Similarity Calculation
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_book_recommendations(title, cosine_sim=cosine_sim):
    # Find exact matches first
    exact_matches = df[df['book_title'].str.lower() == title.lower()]

    # If exact matches are found, use the first one
    if not exact_matches.empty:
        idx = exact_matches.index[0]
    else:
        # If no exact match, fallback to partial match
        partial_matches = df[df['book_title'].str.contains(title, case=False)]
        if not partial_matches.empty:
            idx = partial_matches.index[0]
        else:
            return f"Title '{title}' not found in the dataset."

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices and similarity scores
    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Combine the recommended book titles and their similarity scores
    recommended_books_with_scores = list(zip(df['book_title'].iloc[book_indices], similarity_scores))

    # Convert to DataFrame for easier viewing
    recommendations_df = pd.DataFrame(recommended_books_with_scores, columns=['Book Title', 'Similarity Score'])

    return recommendations_df


def calculate_diversity(recommendation_indices, tfidf_matrix):
    if len(recommendation_indices) < 2:
        return 1  # If there's only one book, consider it fully diverse

    # Extract the TF-IDF vectors for the recommended books
    recommendation_vectors = tfidf_matrix[recommendation_indices]

    # Calculate pairwise cosine similarities between the recommended books
    similarities = cosine_similarity(recommendation_vectors)

    # Calculate the average similarity, excluding the diagonal (self-similarity)
    avg_similarity = (similarities.sum() - len(similarities)) / (len(similarities) * (len(similarities) - 1))

    # Diversity score is the inverse of similarity
    diversity_score = 1 - avg_similarity
    return diversity_score


def get_recommendation_indices(recommendations_df):
    # Extract the indices of the recommended books from the original DataFrame
    recommendation_titles = recommendations_df['Book Title'].values
    recommendation_indices = df[df['book_title'].isin(recommendation_titles)].index
    return recommendation_indices


# Evaluate recommendations for Dune
dune_recommendations = get_book_recommendations('Dune')
print(f"Recommendations for 'Dune' recommendations:\n{dune_recommendations}\n")
dune_recommendation_indices = get_recommendation_indices(dune_recommendations)
dune_diversity_score = calculate_diversity(dune_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'Dune' recommendations: {dune_diversity_score}\n")

# Evaluate recommendations for Dune 2
dune2_recommendations = get_book_recommendations('Dune Messiah (Dune Chronicles, Book 2)')
print(f"Recommendations for 'Dune Messiah (Dune Chronicles, Book 2)' recommendations:\n{dune2_recommendations}\n")
dune2_recommendation_indices = get_recommendation_indices(dune2_recommendations)
dune2_diversity_score = calculate_diversity(dune2_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'Dune 2' recommendations: {dune2_diversity_score}\n")

# Evaluate recommendations for The Hobbit
thehobbit_recommendations = get_book_recommendations('The Hobbit')
print(f"Recommendations for 'The Hobbit' recommendations:\n{thehobbit_recommendations}\n")
thehobbit_recommendation_indices = get_recommendation_indices(thehobbit_recommendations)
thehobbit_diversity_score = calculate_diversity(thehobbit_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'The Hobbit' recommendations: {thehobbit_diversity_score}\n")

# Evaluate recommendations for Harry Potter and the Sorcerer's Stone (Book 1)
harrypotter1_recommendations = get_book_recommendations('Harry Potter and the Sorcerer\'s Stone (Book 1)')
print(f"Recommendations for 'Harry Potter and the Sorcerer\'s Stone (Book 1)' recommendations:\n{harrypotter1_recommendations}\n")
harrypotter1_recommendation_indices = get_recommendation_indices(harrypotter1_recommendations)
harrypotter1_diversity_score = calculate_diversity(harrypotter1_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'Harry Potter and the Sorcerer\'s Stone (Book 1)' recommendations: {harrypotter1_diversity_score}\n")

# Evaluate recommendations for The Testament
thetestament_recommendations = get_book_recommendations('The Testament')
print(f"Recommendations for 'The Testament' recommendations:\n{thetestament_recommendations}\n")
thetestament_recommendation_indices = get_recommendation_indices(thetestament_recommendations)
thetestament_diversity_score = calculate_diversity(thetestament_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'The Testament' recommendations: {thetestament_diversity_score}\n")

# Evaluate recommendations for Animal Farm
animalfarm_recommendations = get_book_recommendations('Animal Farm')
print(f"Recommendations for 'Animal Farm' recommendations:\n{animalfarm_recommendations}\n")
animalfarm_recommendation_indices = get_recommendation_indices(animalfarm_recommendations)
animalfarm_diversity_score = calculate_diversity(animalfarm_recommendation_indices, tfidf_matrix)
print(f"Diversity score for 'Animal Farm' recommendations: {animalfarm_diversity_score}\n")


###########################
#### Additional tests: ####
###########################

# Test the recommendation function on The Hunger Games (Not in Database)
thehungergames_recommendations = get_book_recommendations('The Hunger Games')
print(f"Recommendations for 'The Hunger Games':\n{thehungergames_recommendations}\n")

# Test the recommendation function on Harry Potter (not exact name)
harrypotter_recommendations = get_book_recommendations('Harry Potter')
print(f"Recommendations for 'Harry Potter':\n{harrypotter_recommendations}\n")

# Test the recommendation function on a fake book
fakebookname_recommendations = get_book_recommendations('Fake Book Name')
print(f"Recommendations for 'Fake Book Name':\n{fakebookname_recommendations}\n")
