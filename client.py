import tritonclient.http as httpclient
import numpy as np


def get_embedding(texts, url="localhost:8000", model_name="ensemble_model"):
    client = httpclient.InferenceServerClient(url=url)

    if isinstance(texts, str):
        texts = [texts]
    text_bytes = np.array(texts, dtype=object)  # dtype=object dla stringów

    inputs = []
    inputs.append(httpclient.InferInput("TEXT", text_bytes.shape, "BYTES"))
    inputs[0].set_data_from_numpy(text_bytes)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("EMBEDDING"))

    # Wywołujemy model ensemble o nazwie "ensemble_model"
    result = client.infer(model_name, inputs, outputs=outputs)
    embeddings = result.as_numpy("EMBEDDING")
    return embeddings


# Przykładowe wywołanie
if __name__ == "__main__":
    text1 = "To jest przykładowy tekst"
    text2 = "To jest drugi, zupełnie inny tekst"
    # emb = get_embedding([text1, text2])  # lub [[text1], [text2]], w zależności od kształtu
    emb = get_embedding([[text1], [text2]])  # lub [[text1], [text2]], w zależności od kształtu
    print("Embedding shape:", emb.shape)
    print("Embedding:", emb)