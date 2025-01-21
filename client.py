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
    text2 = "To jest drugi, zupełnie inny tekst i sporo dłuższy tekst w języku polskim, przy którym sprytne modele będą musiały kombinować, w jaki sposób skompresować go w pewien wektor. Mam nadzieję, że pójdzie to dobrze."
    start_time = time()
    for _ in range(100):
        emb = get_embedding([[text1], [text2], [text1*2], [text2*2], [text1*3], [text2*3], [text1*4], [text2*4]])
    end_time = time()
    print(f"Calculation time: {end_time-start_time}")
    print("Embedding shape:", emb.shape)
    print("Embedding:", emb)