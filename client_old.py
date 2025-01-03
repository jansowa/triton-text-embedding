import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *
import argparse
from transformers import AutoTokenizer

def main():
    url = "localhost:8000"
    # client = httpclient.InferenceServerClient(url=url)
    text = "Tutaj będzie jakiś całkiem fajny tekst do pierwszego testu embeddingów :)"
    tokenizer = AutoTokenizer.from_pretrained("sdadas/mmlw-retrieval-roberta-large")
    inputs = tokenizer([text], padding='max_length', truncation=True, max_length=256, return_tensors="np")
    # input_tensors = [httpclient.InferInput("input_ids", inputs['input_ids'].shape, datatype="FP32")]
    # input_tensors[0].set_data_from_numpy(inputs['input_ids'])

    # Połącz się z serwerem Triton
    client = httpclient.InferenceServerClient(url=url)

    # Przygotuj dane wejściowe
    input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT32")
    input_ids.set_data_from_numpy(inputs["input_ids"])

    attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT32")
    attention_mask.set_data_from_numpy(inputs["attention_mask"])

    # Przygotuj dane wyjściowe
    output = httpclient.InferRequestedOutput("last_hidden_state")
    # Wyślij zapytanie do serwera
    response = client.infer(
        model_name="dense_model",
        inputs=[input_ids, attention_mask],
        outputs=[output]
    )

    # Pobierz wyniki
    embedding = response.as_numpy("last_hidden_state")
    print("Embedding shape:", embedding.shape)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_name", default="Select between enemble_model and python_vit"
    # )
    # args = parser.parse_args()
    main()