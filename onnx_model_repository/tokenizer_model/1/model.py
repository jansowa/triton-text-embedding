from transformers import AutoTokenizer

from dense_tokenizer import tokenize

class TritonPythonModel:
    def initialize(self, args):
        model_name = "sdadas/mmlw-retrieval-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def execute(self, requests):
        responses = []
        for request in requests:
            tokenize(self, request, responses)

        return responses