import numpy as np

import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

warnings.filterwarnings('ignore')

columns_name = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('ml-100k/u.data',sep = '\t',names = columns_name)

movie_title = pd.read_csv('ml-100k/u.item',sep = "\|",header = None)
movie_title.head()


movie_title = movie_title[[0,1]]
movie_title.columns = ['item_id','title']

df = pd.merge(df,movie_title,on="item_id")

new =  df

df.groupby('title').mean()['rating'].sort_values(ascending = False).head()
df.groupby('title').count()['rating'].sort_values(ascending = False)

ratings = pd.DataFrame(df.groupby('title').mean()['rating'])

ratings['number of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

ratings.sort_values(by = 'rating',ascending = False)


moviemat = df.pivot_table(index = 'user_id',columns = 'title',values = "rating")

ratings.sort_values('number of ratings',ascending = False).head()

star_wars_user_rating = moviemat['Star Wars (1977)']

star_wars_user_rating.head()

# we have to co relate this starwars with moviemat
similar_to_starwars = moviemat.corrwith(star_wars_user_rating)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['corelation'])

corr_starwars.dropna(inplace=True)

corr_starwars.sort_values('corelation',ascending = False).head(10)

corr_starwars = corr_starwars.join(ratings['number of ratings'])

corr_starwars[corr_starwars['number of ratings'] > 100].sort_values('corelation',ascending = False)

def predict(movie_name):
    movie_user_rating = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_rating)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['corelation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['number of ratings'])
    
    predection = corr_movie[corr_movie['number of ratings'] > 100].sort_values('corelation',ascending = False)
    
    return predection

pred = predict('Titanic (1997)')
pred.head()
