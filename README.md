# My Content : MVP de Recommandation d'Articles

<div class='img'>
  <img src='https://images.unsplash.com/photo-1523995462485-3d171b5c8fa9?q=80&w=1635&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D', alt='papers'>
</div>  

## Introduction

My Content est une start-up innovante qui souhaite révolutionner la façon dont les gens découvrent et consomment du contenu. Notre mission est de faciliter l'accès à la lecture en recommandant des articles et des livres pertinents à chaque utilisateur.

Ce projet GitHub présente le développement d'un Minimum Viable Product (MVP) pour notre application de recommandation. Nous utiliserons des données publiques pour créer un premier prototype fonctionnel, démontrant la valeur de notre concept et permettant de recueillir des retours utilisateurs précieux.

## Objectifs du MVP

* **Fonctionnalité principale :** Recommander 5 articles pertinents à chaque utilisateur.
* **Architecture flexible :** Anticiper l'ajout de nouveaux utilisateurs et de nouveaux articles.
* **Validation du concept :** Tester notre approche de recommandation auprès d'utilisateurs réels.

## Méthodologie

1. **Analyse des données:** Exploration approfondie du jeu de données pour comprendre les informations disponibles et définir les objectifs de recommandation.
2. **Choix de l'algorithme:** Sélection d'un algorithme adapté aux données et aux objectifs (popularité, filtrage collaboratif, etc.).
3. **Développement de l'Azure Function:** Implémentation de l'algorithme de recommandation en Python, intégration avec Azure Blob Storage.
4. **Optimisation des embeddings:** Réduction de dimensionnalité si nécessaire.
5. **Création de l'interface web:** Développement d'une interface simple pour tester l'Azure Function.

### Livrables
- Azure Functions : [Fichier](https://github.com/Zaccaria-Amillou/ociap10/tree/main/azure_functions)
- EDA : [Notebook](https://github.com/Zaccaria-Amillou/ociap10/blob/main/notebook/analyse.ipynb)
- Modélisation : [Notebook](https://github.com/Zaccaria-Amillou/ociap10/blob/main/notebook/modelisation.ipynb)
