import unittest
import pandas as pd
import numpy as np
from recommendation.utils import recommend_articles, collaborativeFilteringRecommendArticle, hybrid_recommend_articles

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Création d'un DataFrame d'articles et de clics pour les tests
        self.articles = pd.DataFrame(np.random.rand(10, 5))
        self.clicks = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'click_article_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        self.user_id = 1

    def test_recommend_articles(self):
        # Test de la fonction recommend_articles
        recommendations = recommend_articles(self.articles, self.clicks, self.user_id)
        print("Recommandations de recommend_articles:", recommendations)
        # Vérification que la fonction retourne bien 5 recommandations
        self.assertEqual(len(recommendations), 5)

    def test_collaborativeFilteringRecommendArticle(self):
        # Test de la fonction collaborativeFilteringRecommendArticle
        recommendations = collaborativeFilteringRecommendArticle(self.clicks, self.user_id)
        print("Recommandations de collaborativeFilteringRecommendArticle:", recommendations)
        # Vérification que la fonction retourne bien 5 recommandations
        self.assertEqual(len(recommendations), 5)
        # Vérification que l'utilisateur n'est pas dans les recommandations
        self.assertNotIn(self.user_id, recommendations)

    def test_hybrid_recommend_articles(self):
        # Test de la fonction hybrid_recommend_articles
        recommendations = hybrid_recommend_articles(self.articles, self.clicks, self.user_id)
        print("Recommandations de hybrid_recommend_articles:", recommendations)
        # Vérification que la fonction retourne bien 5 recommandations
        self.assertEqual(len(recommendations), 5)
        # Vérification que l'utilisateur n'est pas dans les recommandations
        self.assertNotIn(self.user_id, recommendations)

if __name__ == '__main__':
    # Exécution des tests
    unittest.main()