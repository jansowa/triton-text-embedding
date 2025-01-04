1. Prepare docker container with requirements:
```shell
docker run -it --gpus all -v $(pwd):/app -w /app nvcr.io/nvidia/tritonserver:23.10-py3 bash
pip install -r requirements.txt
```
2. Export model to ONNX format
```shell
optimum-cli export onnx --model sdadas/mmlw-retrieval-roberta-large --task feature-extraction onnx_model_repository/embedding_model/1/
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

4. Run client (check "container_name" with ```docker ps```):
```shell
docker exec -it <container_name> /bin/bash
python3 client.py
```