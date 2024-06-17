from math import floor
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, NormalPredictor
from surprise.model_selection import train_test_split
from heapq import nlargest
from collections import defaultdict
import numpy as np
import pickle
from scipy.sparse import csr_matrix

def recommend_articles(articles, clicks, user_id, n=5):
    """
    Cette fonction recommande des articles basés sur la similarité du contenu.
    
    Paramètres :
    articles : DataFrame contenant les embeddings des articles
    clicks : DataFrame contenant les clics des utilisateurs sur les articles
    user_id : ID de l'utilisateur pour lequel faire des recommandations
    n : Nombre d'articles à recommander (par défaut 5)
    
    Retourne :
    Liste des ID des articles recommandés
    """
    # Conversion des user_id et click_article_id en type entier
    clicks['user_id'] = clicks['user_id'].astype(int)
    clicks['click_article_id'] = clicks['click_article_id'].astype(int)
    articles.index = articles.index.astype(int)
    
    # Récupération des articles lus par l'utilisateur
    articles_read = clicks[clicks['user_id'] == int(user_id)]['click_article_id'].tolist()
    print(f"Articles lus par l'utilisateur {user_id}: {articles_read}")

    # Si l'utilisateur n'a lu aucun article, recommander les plus populaires
    if len(articles_read) == 0:
        most_popular_articles = clicks['click_article_id'].value_counts().index.tolist()
        print(f"L'utilisateur {user_id} n'a lu aucun article. Recommandation des articles les plus populaires: {most_popular_articles[:n]}")
        return most_popular_articles[:n]

    # Récupération des embeddings des articles lus par l'utilisateur
    articles_read_embedding = articles.loc[articles_read]
    print(f"Nombre d'articles lus par l'utilisateur {user_id}: {len(articles_read)}")

    # Suppression des articles lus par l'utilisateur de la liste des articles
    articles = articles.drop(articles_read)
    print(f"Articles restants après suppression des articles lus par l'utilisateur {user_id}: {len(articles)}")

    # Calcul de la similarité cosinus entre les articles lus par l'utilisateur et les autres articles
    matrix = cosine_similarity(articles_read_embedding, articles)

    recommendations = []

    # Recommandation des articles les plus similaires aux articles lus par l'utilisateur
    for i in range(n):
        coord_x = floor(np.argmax(matrix)/matrix.shape[1])
        coord_y = np.argmax(matrix)%matrix.shape[1]

        recommendations.append(int(articles.index[coord_y]))

        # Mise à zéro de la similarité de l'article recommandé
        matrix[coord_x][coord_y] = 0

    # Affichage du nombre d'articles recommandés qui ont déjà été lus par l'utilisateur
    already_read = len(set(recommendations) & set(articles_read))
    print(f"Nombre d'articles recommandés qui ont déjà été lus par l'utilisateur {user_id}: {already_read}")

    return recommendations

def collaborativeFilteringRecommendArticle(clicks, user_id, n=5):
    """
    Cette fonction recommande des articles basés sur le filtrage collaboratif.
    
    Paramètres :
    clicks : DataFrame contenant les clics des utilisateurs sur les articles
    user_id : ID de l'utilisateur pour lequel faire des recommandations
    n : Nombre d'articles à recommander (par défaut 5)
    
    Retourne :
    Liste des ID des articles recommandés
    """
    # Conversion des user_id et click_article_id en type entier
    clicks['user_id'] = clicks['user_id'].astype(int)
    clicks['click_article_id'] = clicks['click_article_id'].astype(int)
    
    # Récupération des articles lus par l'utilisateur cible
    articles_read = clicks[clicks['user_id'] == user_id]['click_article_id'].values
    
    # Récupération des utilisateurs qui ont lu les mêmes articles que l'utilisateur cible
    similar_users = clicks[clicks['click_article_id'].isin(articles_read)]['user_id'].unique()
    
    # Récupération des articles lus par les utilisateurs similaires
    similar_users_articles = clicks[clicks['user_id'].isin(similar_users)]['click_article_id'].unique()
    
    # Suppression des articles déjà lus par l'utilisateur cible
    recommendations = [article for article in similar_users_articles if article not in articles_read]
    
    # Si il n'y a pas assez de recommandations, ajout des articles les plus populaires
    if len(recommendations) < n:
        most_popular_articles = clicks['click_article_id'].value_counts().index.tolist()
        for article in most_popular_articles:
            if article not in recommendations and article not in articles_read:
                recommendations.append(article)
            if len(recommendations) == n:
                break
    
    return recommendations[:n]


def hybrid_recommend_articles(articles, clicks, user_id, n=5):
    """
    Cette fonction recommande des articles basés sur une approche hybride (contenu + filtrage collaboratif).
    
    Paramètres :
    articles : DataFrame contenant les embeddings des articles
    clicks : DataFrame contenant les clics des utilisateurs sur les articles
    user_id : ID de l'utilisateur pour lequel faire des recommandations
    n : Nombre d'articles à recommander (par défaut 5)
    
    Retourne :
    Liste des ID des articles recommandés
    """
    # Conversion des user_id et click_article_id en type entier
    clicks['user_id'] = clicks['user_id'].astype(int)
    clicks['click_article_id'] = clicks['click_article_id'].astype(int)
    articles.index = articles.index.astype(int)
    
    # Récupération des articles lus par l'utilisateur cible
    articles_read = clicks[clicks['user_id'] == user_id]['click_article_id'].values
    
    # Récupération des embeddings des articles lus par l'utilisateur
    articles_read_embedding = articles.loc[articles_read]
    
    # Suppression des articles lus par l'utilisateur de la liste des articles
    articles = articles.drop(articles_read)
    
    # Calcul de la similarité cosinus entre les articles lus par l'utilisateur et les autres articles
    matrix = cosine_similarity(articles_read_embedding, articles)
    
    # Recommandation des articles les plus similaires aux articles lus par l'utilisateur
    content_based_recommendations = [int(articles.index[np.argmax(row)]) for row in matrix]
    
    # Récupération des utilisateurs qui ont lu les mêmes articles que l'utilisateur cible
    similar_users = clicks[clicks['click_article_id'].isin(articles_read)]['user_id'].unique()
    
    # Récupération des articles lus par les utilisateurs similaires
    similar_users_articles = clicks[clicks['user_id'].isin(similar_users)]['click_article_id'].unique()
    
    # Suppression des articles déjà lus par l'utilisateur cible
    collaborative_filtering_recommendations = [article for article in similar_users_articles if article not in articles_read]
    
    # Combinaison des recommandations des approches basées sur le contenu et le filtrage collaboratif
    recommendations = content_based_recommendations + collaborative_filtering_recommendations
    
    # Si il n'y a pas assez de recommandations, ajout des articles les plus populaires
    if len(recommendations) < n:
        most_popular_articles = clicks['click_article_id'].value_counts().index.tolist()
        for article in most_popular_articles:
            if article not in recommendations and article not in articles_read:
                recommendations.append(article)
            if len(recommendations) == n:
                break
    
    return recommendations[:n]