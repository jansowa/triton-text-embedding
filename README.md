to run container:

docker run -it --gpus all -v $(pwd):/app -w /app nvcr.io/nvidia/tritonserver:23.10-py3 bash
pip install -r model_repository/dense_model/1/requirements.txt 
tritonserver --model-repository=model_repository --log-verbose=1

Uruchamiamy ten sam kontener poprzez:
docker exec -it <nazwa_kontenera> /bin/bash
następnie:
pip install tritonclient[all]
python3 client.py