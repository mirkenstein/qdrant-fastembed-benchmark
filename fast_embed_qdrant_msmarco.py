from qdrant_client import QdrantClient
import datetime
import time
import csv

# https://github.com/mayya-sharipova/msmarco/blob/main/msmarco-passagetest2019-unique.tsv
data_dir = "./"
csv_file = data_dir + "msmarco-passagetest2019-unique.tsv"
csv_file = open(csv_file, mode='r')

# Read file
id_list = []
text_list = []
for id_col, text_col in csv.reader(csv_file, delimiter='\t'):
    id_list.append(int(id_col))
    text_list.append(text_col)


# Set  client connection string
def set_client(client_type):
    match client_type:
        case "memory":
            return QdrantClient(":memory:")
        case "rest":
            return QdrantClient(url="http://localhost:6333")
        case "grpc":
            return QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        case _:
            return QdrantClient(":memory:")


client = set_client("grpc")

# Set the default embedding model one of
for model_size in ["small", "base", "large"]:
    client.delete_collection("msmarco")
    client.DEFAULT_EMBEDDING_MODEL = "BAAI/bge-" + model_size + "-en-v1.5"
    print("Running with model: BAAI/bge-" + model_size + "-en-v1.5")
    client.set_model(client.DEFAULT_EMBEDDING_MODEL, providers=["CUDAExecutionProvider"])

    print(datetime.datetime.now())
    start_time = time.time()
    client.add(
        collection_name="msmarco",
        documents=text_list,
        ids=id_list
    )
    print(datetime.datetime.now())
    print("--- Model->%s completed task in %s seconds ---" % (model_size, time.time() - start_time))
