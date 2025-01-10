import torch
import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Nazwa modelu z HuggingFace
        model_name = "sdadas/mmlw-retrieval-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Pobranie wejścia
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw_texts = in_tensor.as_numpy()  # np.array obiektów typu bytes/string
            texts = [
                x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
                for x in raw_texts.reshape(-1)
            ]

            # Tokenizacja (bez samego modelu transformera)
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128
            )

            # Zamiana słowników tokenizer-a na numpy int64
            input_ids = np.array(encoded["input_ids"], dtype=np.int32)
            attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)

            # Przygotowanie tensorów wyjściowych
            out_ids_tensor = pb_utils.Tensor("INPUT_IDS", input_ids)
            out_mask_tensor = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            # Odpowiedź
            responses.append(pb_utils.InferenceResponse([
                out_ids_tensor,
                out_mask_tensor
            ]))

        return responses