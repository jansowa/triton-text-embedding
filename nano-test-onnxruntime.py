from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sdadas/mmlw-retrieval-roberta-large")

queries = ["zapytanie: Jak dożyć 100 lat?"]
inputs = tokenizer(queries, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

import numpy as np
# Zamiana tensora torch na numpy
input_ids_np = input_ids.cpu().numpy().astype(np.int64)  # int32 dla tokenów
attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)



import onnxruntime as ort

# Włącz TensorRT Execution Provider:
sess_options = ort.SessionOptions()
session = ort.InferenceSession("onnx/model.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

inputs_onnx = {
    "input_ids": input_ids_np,
    "attention_mask": attention_mask_np
}

outputs = session.run(None, inputs_onnx)

print(outputs.shape)

print(outputs)
