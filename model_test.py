from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sdadas/mmlw-retrieval-roberta-large")

queries = ["zapytanie: Jak dożyć 100 lat?"]
inputs = tokenizer(queries, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open("model.plan", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

print("Drukowanie nazw i ksztltow wejsc/wyjsc silnika")
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    shape = engine.get_binding_shape(i)
    print(i, name, dtype, shape)
print("Koniec drukowania")


import torch
import numpy as np

# Zamiana tensora torch na numpy
input_ids_np = input_ids.cpu().numpy().astype(np.int32)  # int32 dla tokenów
attention_mask_np = attention_mask.cpu().numpy().astype(np.int32)

# Znajdź indeksy wejść/wyjść
input_ids_idx = engine.get_binding_index("input_ids")
attention_mask_idx = engine.get_binding_index("attention_mask")
output_idx = engine.get_binding_index("output_0")  # Załóżmy, że takie jest wyjście.

# Alokacja pamięci na GPU
d_input_ids = cuda.mem_alloc(input_ids_np.nbytes)
d_attention_mask = cuda.mem_alloc(attention_mask_np.nbytes)
# Załóżmy, że wyjście ma rozmiar (1, 1024)
output_size = (1,1024)
output_np = np.empty(output_size, dtype=np.float32)
d_output = cuda.mem_alloc(output_np.nbytes)

# Kopiowanie danych na GPU
cuda.memcpy_htod(d_input_ids, input_ids_np)
cuda.memcpy_htod(d_attention_mask, attention_mask_np)

# Ustawianie kształtów w kontekście (jeśli dynamiczne)
context.set_binding_shape(input_ids_idx, input_ids_np.shape)
context.set_binding_shape(attention_mask_idx, attention_mask_np.shape)

# Wektor z wskaźnikami do bufforów
bindings = [0]*engine.num_bindings
bindings[input_ids_idx] = int(d_input_ids)
bindings[attention_mask_idx] = int(d_attention_mask)
bindings[output_idx] = int(d_output)

# Uruchom inference
context.execute_v2(bindings)

# Pobierz wynik
cuda.memcpy_dtoh(output_np, d_output)
