import torch
import numpy as np
from transformers import AutoModel
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Nazwa modelu z HuggingFace
        model_name = "sdadas/mmlw-retrieval-roberta-large"
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Pobranie tensorów wejściowych (output tokenizera)
            ids_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS")
            mask_tensor = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")

            input_ids = torch.from_numpy(ids_tensor.as_numpy())
            attention_mask = torch.from_numpy(mask_tensor.as_numpy())

            # Przeniesienie na GPU, jeśli jest dostępne
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            # Generowanie embeddingów
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # bsz x seq_len x hidden_size
                last_hidden_state = outputs.last_hidden_state
                # W przypadku Roberta weźmy wektor CLS (pierwszy token)
                cls_embeddings = last_hidden_state[:, 0, :]  # [bsz, 1024]
                cls_embeddings = cls_embeddings.cpu().numpy().astype(np.float32)

            # Tworzymy i zwracamy wynik
            out_tensor = pb_utils.Tensor("EMBEDDING", cls_embeddings)
            responses.append(pb_utils.InferenceResponse([out_tensor]))

        return responses
