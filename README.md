[image1]: assets/ibm_watson_recom.png "image1"
[image2]: assets/svd_train_test.png "image2"


# Recommendations with IBM 
Let's create an Article Recommendation Engine for users on the IBM Watson Studio platform by using real user-article-interaction data.

I analyzed the interactions that users have with articles on the IBM Watson Studio platform. Based on that I made recommendations to them about new articles I think they would like. Below you can see an example of what the dashboard could look like displaying recommended articles on the IBM Watson Platform.

![image1]

You can create your own account to become a part of their community, and get a better understanding of their data by [creating an account on their platform here](https://eu-de.dataplatform.cloud.ibm.com/login?preselect_region=true). This project is part of the Udacity 'Data Scientist' Nanodegree program. Please check this [link](https://www.udacity.com/course/data-scientist-nanodegree--nd025) for more information.

## Main types of recommendation techniques

- ***Knowledge Based*** Recommendations
- ***Collaborative Filtering Based*** Recommendations
- ***Content Based*** Recommendations


   | | ***Knowledge / Rank Based*** Recommendations     | ***Collaborative Filtering Based*** Recommendations    | ***Content Based*** Recommendations |
   | :-------------| :------------- | :------------- | :------------- |
   | ***The idea is to recommend items based on ...*** | explicit knowlege about items which meet user specifications (like item assortment, user preferences, keywords provided by users etc.)| connections between users and items and similar user interests. Use ratings from many users across items in a collaborative way. | similar items to the ones you liked before. Often the content of each item and similarities to other items can be found in item, purpose or genre descriptions.
   | ***Cold Start, issues, requirements*** | No Cold Start (ramp-up) problems, however there are rules for knowledge aquisition     | Strong Cold Start problem. Requires info about the collaboration of user-item interactions.       | Cold Start problem: Often no info about user preferences in the begining. No (or low) cold start problem for item-item relations. However, thorough knowledge of each item in order to find similar items is required.
   | ***Example / Intuition*** | Common for luxury or rare purchases (cars, homes, jewelery). Get back items which fullfill certain criteria (e.g., "the maximum price of the car is X")       | "I liked the new Star Wars Film and I know you like SciFi movies, too. You should go to the cinema."       | You like Start Wars but you do not know Avatar. So let's recommend you Avatar. 
   | ***Similarity measurement***  | No Similarity measurement. Here user provide information about the types of recommendations they would like back. |Similarity Measurement via correlation coefficients, euclidian distance | Similarity Measurement via correlation coefficients, euclidian distance, cosine similarity (similarity matrix), TF-IDF (e.g. in case of filtering out the genre from text)

- Besides these traditional techniques for recommendation you could use techniques based on ***matrix factorization***:
    - Singuar Value Decomposition (***SVD***)
    - Funk - Singular Value Decomposition (***FunkSVD***)

        Singular Value Decomposition of a matrix describes its representation as the product of three special matrices (U-S-Vt). From this representation one can read off the singular values ​​of the matrix. Similar to the eigenvalues, these values characterize properties of the matrix. In case of Recommendations these properties are called ***Latent Factors***. A Latent Factor is not observed in the data directly, but we infer it based on the ratings (interactions) users give to items. Finding how items (like articles, movies, books, etc.) and user relate to Latent Factors is central for making predictions with SVD. 
    
    - The U-matrix: 
        - contains info about how users are related to particular latent factors
        - numbers (ratings / interactions) indicate how each user "feels" about each latent factor
        - n rows -- users
        - k columns -- latent factors
    
    - The V-transpose matrix:
        - contains info about how latent factors are related to items (e.g. articles, movies, books)
        - the higher the value the stronger the relationship
        - in case of movies: e.g. A.I. and WALL E are strongly related to the robot latent feature
        - k rows -- latent factors
        - m columns -- items

    - Sigma matrix:
        - k x k diagonal matrix
        - only diagonal elements are not zero
        - same number of rows and columns as number of latent factors
        - values in the diagonal are always positive and sorted from largest to smallest
        - the diagonal indicated how many latent factors we want to keep
        - first weight is associated with the first latent factor
        - if the the weights are larger, this is an indication that the correponding latent factor is more important to reproduce the ratings of the original user item matrix
        - In case of movies: if a movie is strongly related to dogs, then this latent factor is more important in preticting ratings than using preferences on robots or sadness

    Interesting side-note: Singular value decompositions exist for every matrix.

    SVD - Procedure:
    - ***Sort:*** In case of collected ratings over time
        - Newest data --> for testing set
        - Older data --> for training set
        - This avoids using future data for making predictions on past data
    - ***Split*** data into a training and testing partition
    - ***Fit*** the recommender on the training set
    - ***Evaluate*** the performance on the testing set
    - ***Predict*** ratings (user-item-combination) for every pair
    - ***Compare*** the ratings of a certain user to our predictions
    - ***Understand***: If we do this for every rating in the test set we can understand how well our recommendation engine is working


- ***Business Cases For Recommendations***: 
There are 4 ideas to successfully implement recommendations to drive revenue, which include:
    - Relevance
    - Novelty
    - Serendipity
    - Increased Diversity



## Outline of this project
This is the outline for this project (Outline of Jupyter Notebook). 
### Recomenation Engine Description
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
- [Rank Based Recommendations](#Rank)<br>
- [User-User Based Collaborative Filtering](#User-User)<br>
- [Matrix Factorization](#Matrix-Fact)<br>
- [Extras & Concluding](#conclusions)

### Setup and Links
- [Files in the repo](#Files_in_the_repo)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Exploratory Data Analysis <a name="Exploratory-Data-Analysis"></a>

(Main) DataFrame df: Dataset with 45993 observations and 4 columns

- **Numerical** columns:

	| column_name | type | min | max | number NaN |
	| :-------------  | :-------------  | :-------------  | :-------------  | :-------------  |
	| user_id | int64 | 1 | 5149 | 0 | 
	| interaction | int64 | 1 | 1 | 0 | 


- **Categorical** columns:

	| column_name | type | min | max | number NaN |
	| :-------------  | :-------------  | :-------------  | :-------------  | :-------------  |
	| article_id | float64 | 0.0 | 1444.0 | 0 | 
	| title | object | 0 to life-changing app: new apache systemml api on spark shell | you could be looking at it all wrong | 0 | 


- **Dummy** columns:

	| column_name | type | min | max | number NaN |
	| :-------------  | :-------------  | :-------------  | :-------------  | :-------------  |

- There are ***2 numerical*** (2x int and 0x float) columns
- There are ***2 categorical*** columns
- There are ***0 dummy*** columns
- There are ***0 missing values*** in total in the dataset


- **Median, mean and mode --> Indication for a right skewed distribution**

    - median =  3.0
    - mean =  8.93084693085
    - mode =  0    1
    - 50% of individuals have 3.0 or fewer interactions.
    - The total number of user-article interactions in the dataset is 45993.
    - The maximum number of user-article interactions by any 1 user is 364.
    - The most viewed article in the dataset was viewed 937 times.
    - The article_id of the most viewed article is 1429.0.
    - The number of unique articles that have at least 1 rating 714.
    - The number of unique users in the dataset is 5148.
    - The number of unique articles on the IBM platform 1051.

## Rank Based Recommendations <a name="Rank"></a>
- In this dataset we don't actually have ratings for whether a user liked an article or not. We only know that a user has interacted with an article. In these cases, the popularity of an article can really only be based on how often an article was interacted with. A good (cold start) recommendation is to recommend the top articles ordered with most interactions as the top.

    ```
    def get_top_articles(n, df=df):
    ''' Create a list of the top 'n' article titles 

        INPUTS:
        ------------
            n - (int) the number of top articles to return
            df - (pandas dataframe) df as defined at the top of the notebook 

        OUTPUTS:
        ------------
            top_articles - (list) A list of the top 'n' article titles 
        '''
        # Get a list of the top 'n' article ids
        top_articles_ids = get_top_article_ids(n, df)
        
        # Convert back the string elements in this list to float elements
        top_articles_ids = [float(article_id) for article_id in top_articles_ids]
        
        # Transform those ids to names by using the title column of df
        top_articles = list(df.drop_duplicates(subset=['article_id']).set_index('article_id').loc[top_articles_ids]['title'])
        
        return top_articles # Return the top article titles from df (not df_content)

    def get_top_article_ids(n, df=df):
        ''' Create a list of the top 'n' article ids 
        
            INPUTS:
            ------------
                n - (int) the number of top articles to return
                df - (pandas dataframe) df as defined at the top of the notebook 

            OUTPUTS:
            ------------
                top_articles - (list) A list of the top 'n' article ids 
        '''
        # Get the top articles as a list of id float numbers 
        top_articles = list(df.groupby('article_id').count().sort_values(by='user_id', ascending=False).index[0:n])
        
        # Convert the float elements in this list to string elements 
        top_articles = [str(article_id) for article_id in top_articles]

        return top_articles # Return the top article ids
    ```


## User-User Based Collaborative Filtering <a name="User-User"></a>
- In order to perform  collaborative filtering a user-item matrix is needed. Hence the dataframe df has to be shaped with users as the rows and articles as the columns with the following conditions:
    - Each user should only appear in each row once.
    - Each article should only show up in one column.
    - If a user has interacted with an article, then place a 1 where the user-row meets for that article-column. It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.
    - If a user has not interacted with an item, then place a zero where the user-row meets for that article-column.

    ```
    # create the user-article matrix with 1's and 0's
    def create_user_item_matrix(df):
        ''' Create a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
            an article and a 0 otherwise
        
            INPUTS:
            ------------
                df - pandas dataframe with article_id, title, user_id columns

            OUTPUTS:
            ------------
                user_item - user item matrix 
        '''
        
        # Insert a new column 'interaction' which places 1 in each row 
        df['interaction'] = 1
        
        # Create a user-by-item matrix
        user_item = df.groupby(['user_id', 'article_id'])['interaction'].max().unstack()
        
        # In case of any NaN values isert a 0
        user_item.fillna(0, inplace=True)
        
        return user_item # return the user_item matrix 

    user_item = create_user_item_matrix(df)
    ```

- ***10x10 shortcut of the user-item matrix***:

    |   user_id / article_id  |   0.0 |   2.0 |   4.0 |   8.0 |   9.0 |   12.0 |   14.0 |   15.0 |   16.0 |   18.0 |
    |----------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|
    |         1 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         2 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         3 |     0 |     0 |     0 |     0 |     0 |      1 |      0 |      0 |      0 |      0 |
    |         4 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         5 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         6 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         7 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |
    |         8 |     0 |     0 |     0 |     0 |     0 |      0 |      1 |      0 |      0 |      0 |
    |         9 |     0 |     0 |     0 |     0 |     0 |      0 |      1 |      0 |      1 |      0 |
    |        10 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |


    - Shape of user_item: (5149, 714)
    - Example: The number of articles seen by user 1 = 36.0

- ***Collaborative filtering: Find similar users***
Similar users are found by computing the dot product (cosine similyrity) of every pair of users

    ```
    from collections import Counter
    def find_similar_users(user_id, user_item=user_item, return_sim=False):
        ''' Computes the similarity of every pair of users based on the dot product
        
            INPUTS:
            ------------
                user_id - (int) a user_id
                user_item - (pandas dataframe) matrix of users by articles: 
                            1's when a user has interacted with an article, 0 otherwise

            OUTPUTS:
            ------------
                if return_sim==False:
                    most_similar_users - (list) a list of the users in order from most to least similar
                else: 
                    df_similarity  - (dataframe), similarity dataframe for actual user with user_id 
                                    col1: neighbour ids
                                    col2: similarities                                   
        '''
        
        # get the row for the actual user_id from user_item matrix in the shape (1,714)
        user_1 = np.atleast_2d(user_item.loc[user_id, :])
        
        # build the dot product of user_1 with the transposed user_item matrix (all users)
        # Matrix multiplication shaping: (1x714) x (714x5149) = (1x5149)
        dot_prod = user_1.dot(np.transpose(user_item))
        
        # construct a dictionary, key = user ids, value = similarity with user_id
        similarity = {}
        for i in range(1, user_item.shape[0]+1):
            similarity[i] = dot_prod[0][i-1]
        
        # sort this dicctionary with descending similarity
        c = Counter(similarity)
        similarity_ordered = c.most_common()
        
        # get list of neighbourhood users
        neighbours = [item[0] for item in  similarity_ordered]
        
        
        if return_sim == False:
            # remove actual user with user_id from most_similar_users
            neighbours.remove(user_id)
            most_similar_users = neighbours
            
            return most_similar_users # return a list of the users in order from most to least similar
        
        else:
            # get list of neighbourhood similarities 
            similarities = [item[1] for item in  similarity_ordered]
            df_similarity = pd.DataFrame({'neighbour_id': neighbours, 'similarity': similarities})
            return df_similarity
    ```
- ***Collaborative Filtering: Make Recommendations - Approach 1***
    ```
    def get_article_names(article_ids, df=df):
        ''' Provide a list of article names associated with the list of article ids
        
            INPUTS:
            ------------
                article_ids - (list) a list of article ids
                df - (pandas dataframe) df as defined at the top of the notebook

            OUTPUTS:
            ------------
                article_names - (list) a list of article names associated with the list of article ids 
                                (this is identified by the title column)
        '''
        # A list of article names associated with the list of article ids 
        article_names = list(set(list(df[df['article_id'].astype(str).isin(article_ids)]['title'])))
        
        return article_names # Return the article names associated with list of article ids


    def get_user_articles(user_id, user_item=user_item):
        ''' Provide a list of article ids and names associated with the list of article ids
            INPUTS:
            ------------
                user_id - (int) a user id
                user_item - (pandas dataframe) matrix of users by articles: 
                            1's when a user has interacted with an article, 0 otherwise

            OUTPUTS:
            ------------
                article_ids - (list) a list of the article ids seen by the user
                article_names - (list) a list of article names associated with the list of article ids 
                                (this is identified by the doc_full_name column in df_content)

                Description:
                Provides a list of the article_ids and article titles that have been seen by a user
        '''
        # Get a list of the article ids seen by the user. Provide each element of this list as a string value 
        article_ids = user_item.loc[user_id][user_item.loc[user_id] != 0].index.astype(str).tolist()
        
        # By using thgis list of ids get the names of the articles (list)
        article_names = get_article_names(article_ids)
        
        return article_ids, article_names # return the ids and names
    ```

- ***Collaborative Filtering: Make Recommendations - Optimized Approach***
    ```
    def get_top_sorted_users(user_id, df=df, user_item=user_item):
        ''' Generate a DataFrame for the actual user with his neighbor_ids, corresponding similarities 
            and the number of neighbour-article-interactions 
        
            INPUTS:
            ------------
                user_id - (int)
                df - (pandas dataframe) df as defined at the top of the notebook 
                user_item - (pandas dataframe) matrix of users by articles: 
                        1's when a user has interacted with an article, 0 otherwise


            OUTPUTS:
            ------------
                neighbors_df - (pandas dataframe) a dataframe with:
                                neighbor_id - is a neighbor user_id
                                similarity - measure of the similarity of each user to the provided user_id
                                num_interactions - the number of articles viewed by the user - if a u

                Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                                highest of each is higher in the dataframe
        
        '''
        # Construct the neighbours dataframe by using find_similar_users and 
        # set return_sim to True --> to get back a 'similar neighbour dataframe' 
        neighbors_df = find_similar_users(user_id, user_item=user_item, return_sim=True)
        
        # Calculate the number of each neighbour by using a lambda function approach
        neighbors_df['num_interactions'] = neighbors_df['neighbour_id'].apply(lambda x: num_interactions_df[num_interactions_df.index==x]['num_interactions'].values[0])
        
        # Sort this dataframe first by the col 'similarity' and second by the col 'num_interactions'
        neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)
        
        # Remove the actual user with user_id from that dataframe
        neighbors_df = neighbors_df[neighbors_df['neighbour_id'] != user_id]
        return neighbors_df # Return the dataframe specified in the doc_string


    def user_user_recs_part2(user_id, m=10, top_most=True):
        '''
            INPUTS:
            ------------
                user_id - (int) a user id
                m - (int) the number of recommendations you want for the user
                top_most - (bool) if True --> keep for the (numpy array) new_recs 
                            only the intersection with the top_articles numpy array

            OUTPUTS:
            ------------
                recs - (list) a list of recommendations for the user by article id
                rec_names - (list) a list of recommendations for the user by article title

                Description:
                Loops through the users based on closeness to the input user_id
                For each user - finds articles the user hasn't seen before and provides them as recs
                Does this until m recommendations are found

                Notes:
                * Choose the users that have the most total article interactions 
                before choosing those with fewer article interactions.

                * Choose articles with the articles with the most total interactions 
                before choosing those with fewer total interactions. 
    
        '''
        
        # articles_seen by user (we don't want to recommend these)
        articles_seen_ids, articles_seen_names = get_user_articles(user_id)
        
        # Get the neighbours stored in the neighbors_df dataframe
        neighbors_df = get_top_sorted_users(user_id)
        
        # Get the neighbours as list of ids
        closest_neighbors = neighbors_df['neighbour_id'].tolist()
        
        # Get the top 100 articles as a list of ids
        top_articles = get_top_article_ids(100)
        
        #print('top_articles')
        #print(top_articles)
        
        # Keep the recommended articles here
        recs = np.array([])
        
        # Go through the neighbors and identify articles they like the user hasn't seen
        for neighbor in closest_neighbors:
            
            # get article ids and names as lists (of neighbours)
            neighbs_likes_ids, neighbs_likes_names  = get_user_articles(neighbor)
            
            # Obtain recommendations for the actual neighbor
            new_recs = np.setdiff1d(neighbs_likes_ids, articles_seen_ids, assume_unique=True)
            
            if top_most == True:
                # keep only articles of this neighbour which are in th top 100 article list
                new_recs = np.intersect1d(new_recs, top_articles)
            
            # Update recs with new recs
            recs = np.concatenate([recs, new_recs], axis=0)
            
            # Keep unique elements after concatenation
            _, idx = np.unique(recs, return_index=True)
            
            # Keep order (first neighbour is the most similar - so prefer his suggestions)
            recs = recs[np.sort(idx)]
            
            # If we have enough recommendations exit the loop
            if len(recs) > m-1:
                break
        
        # Pull article titles using article ids, keep only m ones
        rec_ids= list(recs)[:m]
        
        # Transform those ids to article names
        rec_names = get_article_names(rec_ids)

        return rec_ids, rec_names  # return your recommendations for this user_id    
    ```
- ***Short check of first neighbors_df for the user with user_id = 131***

    |    |   neighbour_id |   similarity |   num_interactions |
    |---:|---------------:|-------------:|-------------------:|
    |  1 |           3864 |           11 |                 12 |
    |  2 |            912 |            8 |                102 |
    |  3 |           3540 |            8 |                101 |
    |  4 |             98 |            7 |                170 |
    |  5 |           3764 |            6 |                169 |
    |  6 |             23 |            5 |                364 |
    | 11 |           3782 |            5 |                363 |
    |  7 |            203 |            5 |                160 |
    | 12 |           4459 |            5 |                158 |
    |  8 |            371 |            5 |                 95 |

## Matrix Factorization <a name="Content-Recs"></a>
- Now let's use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
- ***Create a Train and Test Split of the user-item-matrix***:
    ```
    df_train = df.head(40000)
    df_test = df.tail(5993)

    def create_test_and_train_user_item(df_train, df_test):
        '''
            INPUTS:
            -----------
                df_train - training dataframe
                df_test - test dataframe

            OUTPUTS:
            ------------
                user_item_train - a user-item matrix of the training dataframe 
                                (unique users for each row and unique articles for each column)
                user_item_test - a user-item matrix of the testing dataframe 
                                (unique users for each row and unique articles for each column)
                test_idx - all of the test user ids
                test_arts - all of the test article ids
        
        '''
        # Get user_item_train from create_user_item_matrix(df)
        user_item_train = create_user_item_matrix(df_train)
        
        # Get user_item_test from create_user_item_matrix(df)
        user_item_test = create_user_item_matrix(df_test)
        
        # test rows (train user_ids) and colums (test articles) of user_item_test meatrix
        test_idx = user_item_test.index 
        test_arts = user_item_test.columns 
        
        # train rows (train user_ids) and colums (train articles) of user_item_train meatrix
        train_idx = user_item_train.index 
        train_arts = user_item_train.columns 
        
        # common rows and columns between train and test matrix
        common_rows = train_idx.intersection(test_idx)
        common_cols = train_arts.intersection(test_arts)
        
        
        # user_item_test based on common rows and columns
        user_item_test = user_item_test.loc[common_rows, common_cols]
        
        return user_item_train, user_item_test, test_idx, test_arts, common_rows, common_cols

    user_item_train, user_item_test, test_idx, test_arts, common_rows, common_cols = create_test_and_train_user_item(df_train, df_test)
    ```

- ***Check the quality of this SVD approach in making predictions:***
    ```
    num_latent_feats = np.arange(0,700,10)

    row_idxs = user_item_train.index.isin(test_idx)
    col_idxs = user_item_train.columns.isin(test_arts)
    u_test = u_train[row_idxs, :]
    vt_test = vt_train[:, col_idxs]

    train_errors_sum = []
    test_errors_sum = []


    for k in num_latent_feats:
        s_train_lat, u_train_lat, vt_train_lat = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
        u_test_lat, vt_test_lat = u_test[:, :k], vt_test[:k, :]
        
        # dot product:
        user_item_train_preds = np.around(np.dot(np.dot(u_train_lat, s_train_lat), vt_train_lat))
        user_item_test_preds = np.around(np.dot(np.dot(u_test_lat, s_train_lat), vt_test_lat))
    
        # Calculate the error of each prediction with the true value
        diffs_train = np.subtract(user_item_train, user_item_train_preds)
        diffs_test = np.subtract(user_item_test, user_item_test_preds)
        
        # Total Error
        err_train = np.sum(np.sum(np.abs(diffs_train)))
        err_test = np.sum(np.sum(np.abs(diffs_test)))
        
        train_errors_sum.append(err_train)
        test_errors_sum.append(err_test)

    plt.plot(num_latent_feats, 1 - np.array(train_errors_sum)/(user_item_train.shape[0]*user_item_train.shape[1]), label='Train');
    plt.plot(num_latent_feats, 1 - np.array(test_errors_sum)/(user_item_test.shape[0]*user_item_test.shape[1]), label='Test');
    plt.xlabel('Number of Latent Features');
    plt.ylabel('Accuracy');
    plt.title('Accuracy vs. Number of Latent Features');
    plt.legend();
    ```

    ![image2]

# Files in the repo <a name="Files_in_the_repo"></a>

- ***README.md*** - the readme file of this repo
- ***/notebook/Recommendations_with_IBM.ipynb*** - the notebook of this repo containing all the necessary code for IBM Watson article recommendations
- ***/notebook/Recommendations_with_IBM.html*** - The html version of the notebook 'Recommendations_with_IBM.ipynb'
- ***/notebook/user_item_matrix*** - a pickle file containing the data of the user-item matrix 
- ***/notebook/text_for_readme_df.txt*** a file with markdown code for a descriptive table visualization for the dataframe ***df*** used in the notebook 
- ***/notebook/text_for_readme_df_content.txt*** a file with markdown code for a descriptive table visualization for the dataframe ***df_content*** used in the notebook.
- ***/notebook/project_tests.py*** a file for testing notebook outputs
- ***assets*** - a folder with images for the README.
- ***/data/user-item-interactions.csv*** - data for the dataframe df - columns: article_id, title and email
- ***/data/articles_community.csv*** - data for the dataframe df_content - columns: doc_body, doc_description, doc_full_name, doc_status, article_id
     
# Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a Command Line Interface (CLI) under Windows you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Recommendations-with-IBM.git
```

- Change Directory
```
$ cd Recommendation-Engines
```

- Create a new Python environment, e.g. rec_ibm. Inside Git Bash (Terminal) write:
```
$ conda create --name rec_ibm
```

- Activate the installed environment via
```
$ conda activate rec_ibm
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
tabulate = 0.8.7
termcolor
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
Recommendation Engines
* [Essentials of recommendation engines: content-based and collaborative filtering](https://towardsdatascience.com/essentials-of-recommendation-engines-content-based-and-collaborative-filtering-31521c964922)
* [AirBnB uses Embeddings for Recommendations](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
* [Location-Based Recommendation Systems](https://link.springer.com/referenceworkentry/10.1007%2F978-3-319-17885-1_1580)
* [Introduction to Recommender System. Part 1 (Collaborative Filtering, Singular Value Decomposition)](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
- [Rank Based Recommendations](https://github.com/ddhartma/Recommendation-Engines#Knowledge_based_recommendations)
- [User-User Based Collaborative Filtering](https://github.com/ddhartma/Recommendation-Engines#Knowledge_based_recommendations)
- [Matrix Factorization based on traditional SVD (Singular Value Decomposition)](https://github.com/ddhartma/Matrix-Factorization-For-Recommendations#Cold_Start_Problem)
* [Deep learning for recommender systems](https://ebaytech.berlin/deep-learning-for-recommender-systems-48c786a20e1a)
* [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Readme
* [python-tabulate to convert pandas DataFrames to Readme tables](https://pypi.org/project/tabulate/)
