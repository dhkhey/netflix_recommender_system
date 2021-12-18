import numpy as np 
import pandas as pd
from collections import defaultdict
import numpy as np 
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib 
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
import os
import random
from multiprocessing import Manager, Process, Queue, Lock

# The consumer function takes data off of the Queue
def consumer(tasks, print_lock, task_lock, matrix, rating_avg, max_row):
    # Synchronize access to the console
    # with print_lock:
    #     print('Starting consumer => {}'.format(os.getpid()))
     
    # Run indefinitely
    while True:
        task = None

        with task_lock:
            try:
                task = tasks.pop()
            except:
                return
        print(task*10000)
        # do task
        for a in range(10000):
            i = task*10000+a
            if i > max_row:
                break
            rating_sum = sum(matrix[:,i])
            user = index_user[i]
            movies = moviesPerUser[user]
            rating_avg[i] = rating_sum/len(movies)


def cal_cosine_similarity(vector_a,vector_b):
    vector_a = np.array([vector_a])
    vector_b = np.array([vector_b])
    return float(cosine_similarity(vector_a, vector_b))


def prediction(chosen_user, chosen_movie, matrix, ratingMean):

    user_a_index = user_index[chosen_user]
    vector_a = np.zeros(matrix.shape[0])
    # vector_a = np.zeros(len(matrix[:,user_a_index]))

    cosine_similarity_list = []

    print("starting prediction for loops")
    for user in usersPerMovie[chosen_movie]:
        if user_index[chosen_user] != user_index[user]:
            user_b_index = user_index[user]
            vector_b = np.zeros(matrix.shape[0])
            for i in range(matrix.shape[0]):
                if matrix[i,user_a_index] > 0:
                    vector_a[i] = matrix[i,user_a_index]-rating_avg[user_a_index]
                if matrix[i,user_b_index] > 0:
                    vector_b[i] = matrix[i,user_b_index]-rating_avg[user_b_index]
            sim = cal_cosine_similarity(vector_a,vector_b)
            if sim > 0: 
                cosine_similarity_list.append((sim,user))
        print("outtermost for loop iter")
    similarity_sort = sorted(cosine_similarity_list,reverse=True)

    print("done prediction for loops")

    if len(similarity_sort) >= 10:
        similarity_sort = similarity_sort[:10]
    elif 0 < len(similarity_sort) < 10:
        similarity_sort = similarity_sort[:len(similarity_sort)]
    # print(similarity_sort)
    if len(similarity_sort) == 0:
        predicted_rating = ratingMean
    else:
        sort_d = {}
        for e in similarity_sort:
            sort_d[e[1]] = e[0]
        # calcualte weighted rating
        nom = 0
        for u in sort_d:
            nom += movieRatings[u,chosen_movie]*sort_d[u]
        predicted_rating = nom/sum(sort_d.values())

    return predicted_rating


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)
            

if __name__ == '__main__':
    filepath = os.environ.get('tmp_dir', '.')
    # Load data (deserialize)
    with open(f'{filepath}/rating_df.pkl', 'rb') as handle:
        ratings_df = pickle.load(handle)

    print("loaded pkl in")

    # randomly select .1% of the data
    # ratings_df = ratings_df.sample(frac=0.10,ignore_index=True)
    train, test = train_test_split(ratings_df, test_size=0.2, random_state=42, shuffle=True)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("separated into train / test")

    uniqueUsers = list(ratings_df['userId'].unique())
    uniqueMovies = list(ratings_df['movieId'].unique())

    # key: each movie; item: a set of user ids
    moviesPerUser = defaultdict(set)
    # key: each user; item: a set of movie ids
    usersPerMovie = defaultdict(set)
    # key: each movie id; item: corresponding movie title
    movieTitles = {}
    # key: a tuple of user and movie; item: corresponding rating 
    movieRatings = {}

    # start creating dictionaries
    for i in range(ratings_df.shape[0]):
        movie = ratings_df['movieId'][i]
        user = ratings_df['userId'][i]
        moviesPerUser[user].add(movie)
        usersPerMovie[movie].add(user)
        movieTitles[movie] = ratings_df['title'][i]
        movieRatings[user,movie] = ratings_df['rating'][i]

    assert len(moviesPerUser) == len(uniqueUsers), '# of keys in moviesPerUser is not equal to actual unique users'
    assert len(usersPerMovie) == len(movieTitles) == len(uniqueMovies), '# of movies is not equal to actual unique movies'
    assert len(movieRatings) == ratings_df.shape[0]

    print("created dictionaries and asserted they are correct")

    all_movies = set(ratings_df['movieId'].unique())
    all_users = set(ratings_df['userId'].unique())

    # get index by movie id
    movie_index = dict(zip(list(all_movies),list(range(len(all_movies)))))
    # get index by user id 
    user_index = dict(zip(list(all_users),list(range(len(all_users)))))
    # get movie id by index
    index_movie = dict(zip(list(range(len(all_movies))),list(all_movies)))
    # get user id by index
    index_user = dict(zip(list(range(len(all_users))),list(all_users)))

    assert len(movie_index.keys()) == len(uniqueMovies), 'Check the length of movie_index vs. uniqueMovies'
    assert len(user_index.keys()) == len(uniqueUsers), 'Check the length of user_index vs. uniqueUsers'

    print("data re-organizing completed, Assertions passed")

    row, col, data = [],[],[]

    # get unique movies from the chosen train dataframe
    for m in train['movieId'].unique():
        # only get unique users that are in the chosen train dataframe
        users_df = train[train['movieId'] == m]
        users = users_df['userId'].unique()
        i = movie_index[m]
        for u in users:
            row.append(i)
            j = user_index[u]
            col.append(j)
            data.append(movieRatings[u,m])

    assert len(row) == len(col) == len(data), 'matrix row, col or data length are not equal'

    print("data re-organizing completed x2, Assertions passed")

    matrix = coo_matrix((data, (row, col)), shape=(len(movie_index),len(user_index)), dtype=np.int8).toarray()

    print("created matrix, preparing for threading")

    # Create a lock object to synchronize resource access
    print_lock = Lock()
    task_lock = Lock()

    consumers = []

    NUM_THREADS = 4

    manager = Manager()
    tasks = manager.list()
    rating_avg = manager.dict()

    max_row = len(user_index)-1
    index = list(range((ratings_df.shape[0]//10000)+1))
    tasks.extend(index)
    # print(len(tasks))


    # Create consumer processes
    for i in range(NUM_THREADS):
        p = Process(target=consumer, args=(tasks, print_lock, task_lock, matrix, rating_avg, max_row))
            
        p.daemon = False
        consumers.append(p)
    print('launching threads')
    for c in consumers:
        c.start()

    for idx, c in enumerate(consumers):
        c.join()
        print(f'cleared process {idx}')

    print('Parent process moving on...')

    ratingMean = sum(d for d in train['rating']) / train.shape[0]

    predictions = [prediction(test['userId'].iloc[i], test['movieId'].iloc[i],matrix,ratingMean) for i in range(test.shape[0])]

    print("done predictions, doing MSE")
    mse = MSE(predictions, test['rating'].tolist())

    print("done, MSE")

    row = [[len(uniqueMovies),len(uniqueUsers), mse]]
    result_df = pd.DataFrame(row, columns=['num_of_movies','num_of_users','mse'])

    print("Writing CSV")
    timestamp = time.time()
    result_df.to_csv('result_%d.csv'%timestamp,index=False)
    print("file written with name: result_%d.csv"%timestamp)

print("Done, safely exiting program!")
