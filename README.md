Using this repository, you can run and compare the embeddings model hosted using nvidia triton with three different back-ends - PyTorch, ONNX and TensorRT.

[!CAUTION]
The code was tested only on nvidia 550 drivers. There may be compatibility problems with other versions.

0. (Optional) Set minimum/optimum/maximum tokens and batch size for TensorRT. The maximum batch size for all back-ends is also written in the pbtxt files (“max_batch_size: 8”) - change the value if necessary.
1. Install [nvidia drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation), [Docker](https://docs.docker.com/engine/install/), [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. Prepare docker container with requirements:
```shell
docker run -it --gpus all -v $(pwd):/app -w /app --name triton-container nvcr.io/nvidia/tritonserver:24.06-py3 bash
pip install -r requirements.txt
```
3. Export model to ONNX format
```shell
optimum-cli export onnx --model sdadas/mmlw-retrieval-roberta-large --task feature-extraction onnx_model_repository/embedding_model/1/
optimum-cli export onnx --model sdadas/polish-reranker-roberta-v2 --task text-classification onnx_reranker_model_repository/embedding_model/1/
```
4. Convert model with TensorRT
```shell
docker run -it --gpus all -v $(pwd):/app -w /app nvcr.io/nvidia/tensorrt:24.06-py3 bash

MIN_TOKENS=$(python3 -c "import settings; print(settings.MIN_TOKENS)") && \
OPT_TOKENS=$(python3 -c "import settings; print(settings.OPT_TOKENS)") && \
MAX_TOKENS=$(python3 -c "import settings; print(settings.MAX_TOKENS)") && \
MIN_BATCH=$(python3 -c "import settings; print(settings.MIN_BATCH)") && \
OPT_BATCH=$(python3 -c "import settings; print(settings.OPT_BATCH)") && \
MAX_BATCH=$(python3 -c "import settings; print(settings.MAX_BATCH)") && \
trtexec --onnx=onnx_model_repository/embedding_model/1/model.onnx --saveEngine=tensorrt_model_repository/embedding_model/1/model.plan --fp16 \
        --minShapes=input_ids:${MIN_BATCH}x${MIN_TOKENS},attention_mask:${MIN_BATCH}x${MIN_TOKENS} \
        --optShapes=input_ids:${OPT_BATCH}x${OPT_TOKENS},attention_mask:${OPT_BATCH}x${OPT_TOKENS} \
        --maxShapes=input_ids:${MAX_BATCH}x${MAX_TOKENS},attention_mask:${MAX_BATCH}x${MAX_TOKENS}
```
5. Run inference server with one of supported backends:
- Python backend:
```shell
tritonserver --model-repository=model_repository --log-verbose=1
```
- ONNX backend:
```shell
tritonserver --model-repository=onnx_model_repository --log-verbose=1
```
- TensorRT backend:
```shell
tritonserver --model-repository=tensorrt_model_repository --log-verbose=1
```
6. Run client (check "container_name" with ```docker ps```):
```shell
docker exec -it <container_name> /bin/bash
python3 client.py
```