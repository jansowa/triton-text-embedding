import numpy as np

import triton_python_backend_utils as pb_utils
from settings import MAX_TOKENS


def tokenize(triton_python_model, request, responses):
    # Pobranie wejścia
    in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
    raw_texts = in_tensor.as_numpy()  # np.array obiektów typu bytes/string
    texts = [
        x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
        for x in raw_texts.reshape(-1)
    ]
    # Tokenizacja (bez samego modelu transformera)
    encoded = triton_python_model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS
    )
    # Zamiana słowników tokenizer-a na numpy int64
    input_ids = np.array(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.array(encoded["attention_mask"], dtype=np.int64)
    # Przygotowanie tensorów wyjściowych
    out_ids_tensor = pb_utils.Tensor("INPUT_IDS", input_ids)
    out_mask_tensor = pb_utils.Tensor("ATTENTION_MASK", attention_mask)
    # Odpowiedź
    responses.append(pb_utils.InferenceResponse([
        out_ids_tensor,
        out_mask_tensor
    ]))
