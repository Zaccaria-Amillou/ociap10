from collections import defaultdict
from heapq import nlargest
from surprise import Reader, Dataset, SVDpp, train_test_split

class Recommender:
    def __init__(self, articles, clicks):
        self.articles = articles
        self.clicks = clicks

    def precision_recall_at_k(self, predictions, k_list=[5, 10]):
        '''Return precision and recall at k over all users for multiple values of k'''

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = {k: dict() for k in k_list}
        recalls = {k: dict() for k in k_list}
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            for k in k_list:
                # Number of recommended items in top k
                n_rec_k = len(user_ratings[:k])

                # Number of relevant and recommended items in top k
                n_rel_and_rec_k = sum((true_r == est) for (est, true_r) in user_ratings[:k])

                # Precision@K: Proportion of recommended items that are relevant
                precisions[k][uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

                # Number of relevant items
                n_rel = sum((true_r == est) for (est, true_r) in user_ratings)

                # Recall@K: Proportion of relevant items that are recommended
                recalls[k][uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

    def recommend_articles(self, user_id, n=5):
        # Convert user_id and click_article_id to integer type
        self.clicks['user_id'] = self.clicks['user_id'].astype(int)
        self.clicks['click_article_id'] = self.clicks['click_article_id'].astype(int)
        self.articles.index = self.articles.index.astype(int)
        
        # Check if user_id is in clicks
        if user_id not in self.clicks['user_id'].values:
            return f"Error: User ID {user_id} not found in clicks data."

        # Create a new DataFrame that counts the number of times a user clicked on an article
        click_counts = self.clicks.groupby(['user_id', 'click_article_id']).size().reset_index(name='click_count')

        # Use a smaller subset of data for the collaborative filtering to avoid memory issues
        data_subset = click_counts

        # Create a reader and a data object
        reader = Reader(rating_scale=(1, data_subset.click_count.max()))  # assuming a click count of at least 1
        data = Dataset.load_from_df(data_subset, reader)

        # Split the data into train and test sets
        trainset, testset = train_test_split(data, test_size=0.2)

        # Train a SVD model with the best parameters
        algo = SVDpp(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
        algo.fit(trainset)

        # Predict ratings for the testset
        predictions_test = algo.test(testset)

        # Calculate precision and recall at k
        precisions, recalls = self.precision_recall_at_k(predictions_test, k_list=[5, 10])
        for k in [5, 10]:
            avg_precision = sum(prec for prec in precisions[k].values()) / len(precisions[k])
            avg_recall = sum(rec for rec in recalls[k].values()) / len(recalls[k])
            print(f"Average Precision at {k}: {avg_precision}")
            print(f"Average Recall at {k}: {avg_recall}")

        # Get the list of articles read by the user
        articles_read = self.clicks[self.clicks['user_id'] == user_id]['click_article_id'].tolist()

        # Get the list of all articles
        all_articles = list(self.articles.index)

        # Remove the articles already read by the user
        articles_to_predict = [article for article in all_articles if article not in articles_read]

        # Get the predicted ratings for the articles not yet read by the user
        predictions = {article: algo.predict(user_id, article).est for article in articles_to_predict}

        # Get the top n articles
        top_n_articles = nlargest(n, predictions, key=predictions.get)

        return top_n_articles