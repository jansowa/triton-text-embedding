from transformers import AutoModel, AutoTokenizer
model_name = "sdadas/mmlw-retrieval-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
