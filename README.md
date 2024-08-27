# Qdrant Client GPU

Testing [Qdrant client](https://github.com/qdrant/qdrant-client) with [FastEmbed](https://qdrant.github.io/fastembed/).

The script runs over the 3 `BAAI/bge-[small|base|large]-en-v1.5` models for a 
given Qdrant client and indexes a  [MS MARCO ](https://microsoft.github.io/msmarco/) dataset

1. Launch Local Qdrant Server
https://qdrant.tech/documentation/quickstart/

```shell
docker run -p 6333:6333 -p 6334:6334 \
    qdrant/qdrant
```

2. Download Data
Using [MS MARCO ](https://microsoft.github.io/msmarco/) Dataset with 182,469 entries.

Data file used can be downloaded [here](https://github.com/mayya-sharipova/msmarco/blob/main/msmarco-passagetest2019-unique.tsv)
 
Download the file and place it in your `data_dir` location.
 

3. Install `qdrant-client[fastembed-gpu]` and `onnxruntime-gpu`
```shell
pip install 'qdrant-client[fastembed-gpu]'
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

```

4. Run [fast_embed_qdrant_msmarco.py](fast_embed_qdrant_msmarco.py)
Here are the results for the 3  models from the `BAAI/bge` family.
https://qdrant.github.io/fastembed/examples/Supported_Models/

Indexing time for REST and GRPC and Memory Clients to ingest 183K lines of text 62MB large
 
| Model                   | Dims   | REST | GRPC | Memory | Data Size |
|-------------------------|--------|------|------|--------|-----------|
| BAAI/bge-small-en-v1.5  | 384    | 145s | 111s | 97s    | 392MB     |
| BAAI/bge-base-en-v1.5   | 768    | 235s | 189s | 166s   | 646MB     |
| BAAI/bge-large-en-v1.5  | 1024   | 807s | 746s | 726s   | 827MB     | 

### Prerequisites ONNX Runtime and Troubleshooting

Check version requirements https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

To install specific version of `cudnn`
```shell
pip install nvidia-cudnn-cu12==9.3.0.75
```
Consult with the pypi page https://pypi.org/project/nvidia-cudnn-cu12/
Also might need to set your environment variable `LD_LIBRARY_PATH`

 


https://github.com/qdrant/qdrant-client?tab=readme-ov-file#fast-embeddings--simpler-api
> Note: `fastembed-gpu` and `fastembed` are mutually exclusive. You can only install one of them.
>
> If you previously installed `fastembed`, you might need to start from a fresh environment to install `fastembed-gpu`.


Fedora Cuda Repo
https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64

Need to install Cuda Toolkit if not already present on your system.
```shell
dnf install cuda-tools-12-6
dnf install libcufft-devel-12-6
dnf install libcublas-devel-12-6
dnf install libcurand-devel-12-6
```

This would solve issues with errors like
```shell
Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory
```
and 
```shell
Failed to load library libonnxruntime_providers_cuda.so with error: libcudart.so.12: cannot open shared object file: cannot open shared object file:
```
In addition, setting the env variables
```shell
LD_LIBRARY_PATH=".venv/lib/python3.11/site-packages/nvidia/cudnn/lib":${LD_LIBRARY_PATH}
CUDA_HOME=/usr/local/cuda-12.5/
```
Will solve  `ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory`

### Torch with Cuda

Alternatively can use nvidia image from here:
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

```
docker run --gpus all -it --rm -v $HOME/PycharmProjects/torch:/project nvcr.io/nvidia/pytorch:24.07-py3
```