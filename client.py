import tritonclient.http as httpclient
import numpy as np
from time import time

def get_embedding(texts, url="localhost:8000", model_name="ensemble_model"):
    client = httpclient.InferenceServerClient(url=url)

    if isinstance(texts, str):
        texts = [texts]
    text_bytes = np.array(texts, dtype=object)

    inputs = []
    inputs.append(httpclient.InferInput("TEXT", text_bytes.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_bytes)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("EMBEDDING"))

    result = client.infer(model_name, inputs, outputs=outputs)
    embeddings = result.as_numpy("EMBEDDING")
    return embeddings


if __name__ == "__main__":
    text1 = "To jest przykładowy tekst"
    text2 = "To jest drugi, zupełnie inny tekst"
    start_time = time()
    # tensorrt 1.09-1.14
    # tensorrt 0.78-0.83
    # onnx 1.56-1.57
    # python 1.28-1.29
    for _ in range(100):
        emb = get_embedding([[text1], [text2], [text1], [text2], [text1], [text2], [text1], [text2]])
    end_time = time()
    print(f"Calculation time: {end_time-start_time}")
    print("Embedding shape:", emb.shape)
    print("Embedding:", emb)