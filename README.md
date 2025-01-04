To run inference server:
```shell
docker run -it --gpus all -v $(pwd):/app -w /app nvcr.io/nvidia/tritonserver:23.10-py3 bash
pip install -r model_repository/dense_model/1/requirements.txt 
tritonserver --model-repository=model_repository --log-verbose=1
```

To run client (check "container_name" with ```docker ps```):
```shell
docker exec -it <container_name> /bin/bash
następnie:
pip install tritonclient[all]
python3 client.py
```