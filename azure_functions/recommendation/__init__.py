import logging
import os
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobClient,  __version__
import utils
import io

try:
    logging.info("Azure Blob Storage v" + __version__)

    connect_str = os.getenv('AzureWebJobsStorage')

    ##### Chargement des fichiers #####
    # Create blob clients
    articles_blob = BlobClient.from_connection_string(conn_str=connect_str, container_name="reco-files", blob_name='embeddings_df_pca.csv')
    clicks_blob = BlobClient.from_connection_string(conn_str=connect_str, container_name="reco-files", blob_name='clicks_df.csv')

    # Download blobs and convert to pandas DataFrame
    articles_df = pd.read_csv(io.BytesIO(articles_blob.download_blob().readall()))
    clicks_df = pd.read_csv(io.BytesIO(clicks_blob.download_blob().readall()))

except Exception as ex:
    print('Exception:')
    print(ex)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        id = req_body.get('id')
        type = req_body.get('type')

    if isinstance(id, int) and isinstance(type, str):

        recommended = utils.recommend_articles(articles_df, clicks_df, id) if type == "cb" else utils.collaborativeFilteringRecommendArticle(articles_df, clicks_df, id)

        return func.HttpResponse(str(recommended), status_code=200)

    else:
        return func.HttpResponse(
             "Requête invalide.\nDans le body doit figurer sous format json :\n- id (int)\n- type (cb ou cf)",
             status_code=400
        )