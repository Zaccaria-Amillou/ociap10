import logging
import os
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobClient,  __version__
from . import utils
import io
import pickle


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        logging.info("Azure Blob Storage v" + __version__)

        connect_str = os.getenv('AzureWebJobsStorage')

        ##### Chargement des fichiers #####
        # Download blobs
        blob_articles = BlobClient.from_connection_string(conn_str=connect_str, container_name="reco-files", blob_name='embeddings_df_pca.pkl')
        blob_users = BlobClient.from_connection_string(conn_str=connect_str, container_name="reco-files", blob_name='clicks_df.pkl')
        # Load to pickle
        articles_df = pickle.loads(blob_articles.download_blob().readall())
        clicks_df = pickle.loads(blob_users.download_blob().readall())

    except Exception as ex:
        print('Exception:')
        print(ex)
        return func.HttpResponse(
             "Error loading data from Azure Blob Storage",
             status_code=500
        )

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        id = req_body.get('id')
        type = req_body.get('type')

    if isinstance(id, int) and isinstance(type, str):

        if type == "ra":
            recommended = utils.recommend_articles(articles_df, clicks_df, id)
        elif type == "cf":
            recommended = utils.collaborativeFilteringRecommendArticle(clicks_df, id)
        elif type == "hy":
            recommended = utils.hybrid_recommend_articles(articles_df,clicks_df, id)

        return func.HttpResponse(str(recommended), status_code=200)

    else:
        return func.HttpResponse(
             "Requête invalide.\nDans le body doit figurer sous format json :\n- id (int)\n- type (cb ou cf)",
             status_code=400
        )