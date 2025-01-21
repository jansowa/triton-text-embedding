1. Prepare docker container with requirements:
```shell
docker run -it --gpus all -v $(pwd):/app -w /app --name triton-container nvcr.io/nvidia/tritonserver:24.06-py3 bash
pip install -r requirements.txt
```
2. Export model to ONNX format
```shell
optimum-cli export onnx --model sdadas/mmlw-retrieval-roberta-large --task feature-extraction onnx_model_repository/embedding_model/1/
optimum-cli export onnx --model sdadas/polish-reranker-roberta-v2 --task text-classification onnx_reranker_model_repository/embedding_model/1/
```

3. Convert model with TensorRT
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

3. Run inference server with one of supported backends:
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

4. Run client (check "container_name" with ```docker ps```):
```shell
docker exec -it <container_name> /bin/bash
python3 client.py
```