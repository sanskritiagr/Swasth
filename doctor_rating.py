import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data_0 = pd.read_csv('./rating.csv')
doctor_info = pd.read_csv('./doc_table.csv')

def merge_tables(given):
    merged_df = pd.merge(given, doctor_info, on='Doctor Name', how='inner')
    merged_df.sort_values(by='Rating', ascending=False, inplace=True)
    return merged_df

def avg_rating():
    # Calculate mean ratings for each doctor
    mean_ratings = data_0.mean(axis=0)
    mean_ratings_df = pd.DataFrame(mean_ratings, columns=['Rating'])
    mean_ratings_df.reset_index(inplace=True)
    mean_ratings_df.rename(columns={'index': 'Doctor Name'}, inplace=True)
    return mean_ratings_df


def col_filtering(picked_userid):
    # Normalize user-item matrix
    matrix_norm = data_0.subtract(data_0.mean(axis=1), axis = 'rows')

    #Identify Similar Users

    # User similarity matrix using Pearson correlation
    user_similarity = matrix_norm.T.corr()

    # User similarity matrix using cosine similarity
    user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))

    # Number of similar users
    n = 100

    # User similarity threashold
    user_similarity_threshold = 0.1

    # Get top n similar users
    similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

    # Print out top n similar users
    # print(f'The similar users for user {picked_userid} are', similar_users)

    #Narrow Down Item Pool

    # Movies that similar users watched. Remove movies that none of the similar users have watched
    similar_user_docs = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')

    # Recommend Items
    # A dictionary to store item scores
    item_score = {}

    # Loop through items
    for i in similar_user_docs.columns:
        # Get the ratings for movie i
        doc_rating = similar_user_docs[i]
        # Create a variable to store the score
        total = 0
        # Create a variable to store the number of scores
        count = 0
        # Loop through similar users
        for u in similar_users.index:
            # If the doctor has rating
            if pd.isna(doc_rating[u]) == False:
                # Score is the sum of user similarity score multiply by the doctor rating
                score = similar_users[u] * doc_rating[u]
                # Add the score to the total score for the doctor so far
                total += score
                # Add 1 to the count
                count +=1
        # Get the average score for the item
        item_score[i] = total / count

    # Convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['Doctor Name', 'doctor_score'])

    # Sort Doctors by score
    ranked_item_score = item_score.sort_values(by='doctor_score', ascending=False)

    # Select top m movies
    # m = numofdocs

    #Predict Scores

    # Average rating for the picked user
    avg_rating = data_0[data_0.index == picked_userid].T.mean()[picked_userid]

    # Print the average rating for user 
    print(f'The average rating for user {picked_userid} is {avg_rating:.2f}')

    # Calcuate the predicted rating
    ranked_item_score['Rating'] = ranked_item_score['doctor_score'] + avg_rating

    ranked_item_score.drop('doctor_score', axis=1, inplace=True)

    # Take a look at the data
    return ranked_item_score


