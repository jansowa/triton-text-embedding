import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Ładowanie modelu i tokenizera
        model_name = "sdadas/mmlw-retrieval-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # model w trybie ewaluacji
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Pobranie wejścia
            # Załóżmy, że input to string w polu "TEXT"
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            # texts = in_tensor.as_numpy().astype('U')  # Konwersja do tablicy stringów
            # texts = in_tensor.as_numpy()  # Konwersja do tablicy stringów

            raw_texts = in_tensor.as_numpy()
            texts = [
                x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
                for x in raw_texts.reshape(-1)
            ]

            # Tokenizacja
            inputs = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=128)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Zakładamy, że chcemy embeddings z [CLS] (pierwszy token), np. z last hidden state
                # Dla Roberta last_hidden_state ma wymiary [batch_size, seq_len, hidden_size]
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # batch x hidden_size
                cls_embeddings = cls_embeddings.cpu().numpy().astype(np.float32)

            # Tworzymy odpowiedź w formie tensora do Tritona
            out_tensor = pb_utils.Tensor("EMBEDDING", cls_embeddings)
            responses.append(pb_utils.InferenceResponse([out_tensor]))

        return responses