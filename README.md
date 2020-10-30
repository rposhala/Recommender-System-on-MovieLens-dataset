# Recommender-System-on-MovieLens-dataset
## Project Overview
Knowledge-based, Content-based and Collaborative Recommender systems are built on MovieLens dataset with 100,000 movie ratings using Pandas operations and by fitting KNN, SVD & deep learning models along with NLP techniques usage to suggest movies for the users based on similar users and for queries specific to genre, user, movie, rating, popularity.

## Recommender System Overview
A recommender system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. Recommender systems are utilized in a variety of areas including movies, music, news, social tags, and products in general. Recommender systems typically produce a list of recommendations and there are few ways in which it can be done. Two of the most popyular ways are – through collaborative filtering or through content-based filtering.

Most internet products we use today are powered by recommender systems. Youtube, Netflix, Amazon, Pinterest, and long list of other internet products all rely on recommender systems to filter millions of contents and make personalized recommendations to their users. Recommender systems are well-studied and proven to provide tremendous values to internet businesses and their consumers.

There are majorly six types of recommender systems which work primarily in the Media and Entertainment industry: 
- Collaborative Recommender system
- Content-based recommender system
- Knowledge based recommender system
- Hybrid recommender system
- Demographic based recommender system
- Utility based recommender system

Recommender System is a vast concept rooted from a base idea of giving out suggestions to the users. There are wide range of algorithms are used to build a recommender system and the type of recommender system used is mostly dictated by the type of data available. In this project, first three of the above recommender systems were built.

**Content based recommender system** approach utilizes a series of discrete characteristics of an item in order to recommend additional items with similar properties. Content-based filtering methods are based on a description of the item and a profile of the user's preferences. To keep it simple, it will suggest you similar movies based on the movie we give (movie name would be the input) or based on all of the movies watched by a user (user is the input). It extracts features of a item and it can also look at the user's history to make the suggestions.

**Collaborative filtering** is based on the assumption that people who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past. The system generates recommendations using only information about rating profiles for different users or items. By locating peer users/items with a rating history similar to the current user or item, they generate recommendations using this neighborhood. This approach builds a model from a user’s past behaviors (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in. Collaborative filtering methods are classified as memory-based and model-based.

**Knowledge based recommender system**s are based on explicit knowledge about the item assortment, user preferences, and recommendation criteria (i.e., which item should be recommended in which context). These systems are applied in scenarios where alternative approaches such as collaborative filtering and content-based filtering cannot be applied. In simple terms, knowledge based recommender system can be used to suggest content/item to a new user or an anonymous user who doesn't have any history.

**Hybrid recommender system** combines more than one of these techniques to resolve one or more problems. This approach can be used to overcome some of the common problems in recommender systems such as cold start and the sparsity problem in collaborative approach, as well as the knowledge engineering bottleneck in knowledge-based approaches. It is proved that hybrid recommender system performs extremely well compared to pure collaborative and content based methods.

## About Dataset used

[MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) has been used for this project. MovieLens is a rating dataset from the MovieLens website, which has been
collected over some period. Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released on 4/1998. Further information regarding this dataset can be found [here](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt).

A little about the dataset:

MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.
 
This data set consists of:
  - 100,000 ratings (1-5) from 943 users on 1682 movies. 
  - Each user has rated at least 20 movies. 
  - Simple demographic info for the users (age, gender, occupation, zip)

About few components loaded from the package which are used in this project: 

 - u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id | item id | rating | timestamp. 
 - u.info     -- The number of users, items, and ratings in the u data set.
 - u.item     -- Information about the items (movies); this is a tab separated
              list of movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
	      
  The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once.The movie ids are the ones used in the u.data data set.
  - u.genre    -- A list of the genres.

## Data Visualizations and Manipulations

Data has been loaded into dataframes using pandas. It had been analyzed and visualized to draw some key insights before going further into recommendations.


Observation: We can observe that most of the users have rewarded movies they watched with a 4 star rating and followed by 3 star and 5 star.

The same has been displayed below using a pie chart to understand the constributions.

Genre based number of movies count is being plotted using bar-graph:

We can see that most of the movies belong to movie genre : Drama followed by Comedy then Action, Romance and Thriller

#### Dataframes formed and used: 
items_dataset (movie id, movie name, and all genres); dataset (user id, movie id, rating); movie_dataset is a subset of items_dataset (it has movie id, movie name); Both movie_dataset and dataset are merged based on movie id and new merged_dataset is formed (user id, movie id, rating, movie name); a new dataframe is formed by averaging the overall rating available to a movie from the merged_dataset, are sorted with descending order of ratings and is named avg_rating_dataset (movie name, avg rating); 

## Knowledge based Recommender System

Recommendations are made based on the available items and their corresponding ratings data, considering we have no user data available.

Data manipulations are done using **Pandas**

 - A General recommendations of movies made based on high average ratings:
 
 ```
movie title						avg rating
Marlene Dietrich: Shadow and Light (1996)		5.0
Prefontaine (1997)					5.0
Santa with Muscles (1996)				5.0
Star Kid (1997)						5.0
Someone Else's America (1995)				5.0
Entertaining Angels: The Dorothy Day Story (1996)	5.0
Saint of Fort Washington, The (1993)			5.0

 ```
 These are the top 7 movies that can be naviely suggested to the new users, Recommendations based on top average ratings.

 - Movie Recommendations based on popularity :
 
 We have considered movies which have more than 400 viewers as *POPULAR* and there are 12 movies.
 ```
 movie title			Number of Users watched
Star Wars (1977)		583
Contact (1997)			509
Fargo (1996)			508
Return of the Jedi (1983)	507
Liar Liar (1997)		485
English Patient, The (1996)	481
Scream (1996)			478
Toy Story (1995)		452
Air Force One (1997)		431
Independence Day (ID4) (1996)	429
Raiders of the Lost Ark (1981)	420
Godfather, The (1972)		413
```
These are the most popular movies which can be recommended to a new user. *Recommendations based on Popularity*

*Above two recommendations are good enough but are not complete and may not interest many of the new users*

- Movie Recommendations based on both popular and average ratings. 
*Recommendations based popularity and rating.* These are **top rated popular movies**

```
movie title				avg rating	Number of Users watched
Star Wars (1977)			4.358491	583
Silence of the Lambs, The (1991)	4.289744	390
Godfather, The (1972)			4.283293	413
Raiders of the Lost Ark (1981)		4.252381	420
Titanic (1997)				4.245714	350
Empire Strikes Back, The (1980)		4.204360	367
Princess Bride, The (1987)		4.172840	324
Fargo (1996)				4.155512	508
Monty Python and the Holy Grail (1974)	4.066456	316
1Pulp Fiction (1994)			4.060914	394
4Fugitive, The (1993)			4.044643	336
9Return of the Jedi (1983)		4.007890	507
```

These movies are the best to suggest to a new user as they are popular and well rated by the users who already watched them. These have rating more than 4 with atleast 300 viewers. (the threshold for ratings and number of viewers can be changed accordingly based on the data available)

 - Movie suggestions based on specific *Genre* picked by the user
 
 For every genre, Genre wise ratings are plotted using bar plot and the above mentioned three types of movie recommendations are given out in specific to the selected genre.
 
 Below are the two such examples for genres: Action and Children
 
 ```
 ****************************     ****** GENRE:  Action  ******     ******************************
 Total number of users watched this Genre:  25589
  
These are the top movies that can be naviely suggested to the new users for the requested movie genre: Action . Recommendations based on top average ratings.
                                   rating
movie title                              
Star Wars (1977)                 4.358491
Godfather, The (1972)            4.283293
Raiders of the Lost Ark (1981)   4.252381
Titanic (1997)                   4.245714
Empire Strikes Back, The (1980)  4.204360
Boot, Das (1981)                 4.203980
Godfather: Part II, The (1974)   4.186603
African Queen, The (1951)        4.184211
Princess Bride, The (1987)       4.172840
Braveheart (1995)                4.151515
****************************     ******************************     ******************************
These are the most popular movies which can be recommended to a new user in Action genre. Recommendations based on Popularity
                       movie title  Number of Users watched
0                 Star Wars (1977)                      583
1        Return of the Jedi (1983)                      507
2             Air Force One (1997)                      431
3    Independence Day (ID4) (1996)                      429
4   Raiders of the Lost Ark (1981)                      420
5            Godfather, The (1972)                      413
6                 Rock, The (1996)                      378
7  Empire Strikes Back, The (1980)                      367
8  Star Trek: First Contact (1996)                      365
9                   Titanic (1997)                      350
****************************     ******************************     ******************************
These movies are the best to suggest to a new user within their requested genre as they are popular and well rated by the users who already watched them.
These have rating more than  4.0  with atleast  250  viewers.
**Recommendations based popularity and rating. These are top rated popular movies**
                          movie title    rating  Number of Users watched
0                    Star Wars (1977)  4.358491                      583
1               Godfather, The (1972)  4.283293                      413
2      Raiders of the Lost Ark (1981)  4.252381                      420
3                      Titanic (1997)  4.245714                      350
4     Empire Strikes Back, The (1980)  4.204360                      367
8          Princess Bride, The (1987)  4.172840                      324
9                   Braveheart (1995)  4.151515                      297
11               Fugitive, The (1993)  4.044643                      336
12                       Alien (1979)  4.034364                      291
13          Return of the Jedi (1983)  4.007890                      507
14  Terminator 2: Judgment Day (1991)  4.006780                      295
****************************     ******************************     ******************************
```
```
****************************     ****** GENRE:  Children  ******     ******************************
Total number of users watched this Genre:  7182
  
These are the top movies that can be naviely suggested to the new users for the requested movie genre: Children . Recommendations based on top average ratings.
                                               rating
movie title                                          
Star Kid (1997)                              5.000000
Wizard of Oz, The (1939)                     4.077236
Babe (1995)                                  3.995434
Toy Story (1995)                             3.878319
E.T. the Extra-Terrestrial (1982)            3.833333
Aladdin (1992)                               3.812785
Winnie the Pooh and the Blustery Day (1968)  3.800000
Beauty and the Beast (1991)                  3.792079
Lion King, The (1994)                        3.781818
Fantasia (1940)                              3.770115
****************************     ******************************     ******************************
These are the most popular movies which can be recommended to a new user in Children genre. Recommendations based on Popularity
                                    movie title  Number of Users watched
0                              Toy Story (1995)                      452
1  Willy Wonka and the Chocolate Factory (1971)                      326
2             E.T. the Extra-Terrestrial (1982)                      300
3                      Wizard of Oz, The (1939)                      246
4                         Lion King, The (1994)                      220
5                                   Babe (1995)                      219
6                                Aladdin (1992)                      219
7                   Beauty and the Beast (1991)                      202
8                          Fly Away Home (1996)                      180
9                           Mary Poppins (1964)                      178
****************************     ******************************     ******************************
These movies are the best to suggest to a new user within their requested genre as they are popular and well rated by the users who already watched them.
These have rating more than  3.0  with atleast  150  viewers.
**Recommendations based popularity and rating. These are top rated popular movies**
                                     movie title  ...  Number of Users watched
1                       Wizard of Oz, The (1939)  ...                      246
2                                    Babe (1995)  ...                      219
3                               Toy Story (1995)  ...                      452
4              E.T. the Extra-Terrestrial (1982)  ...                      300
5                                 Aladdin (1992)  ...                      219
7                    Beauty and the Beast (1991)  ...                      202
8                          Lion King, The (1994)  ...                      220
9                                Fantasia (1940)  ...                      174
10                           Mary Poppins (1964)  ...                      178
11        Snow White and the Seven Dwarfs (1937)  ...                      172
15  Willy Wonka and the Chocolate Factory (1971)  ...                      326
16                          Fly Away Home (1996)  ...                      180

[12 rows x 3 columns]
****************************     ******************************     ******************************
```
