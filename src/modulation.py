import torch
from rich import print
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
)
from src.interpolation import uniform_slerp
import time

categories = {
    "color": ["red", "green", "blue"],
    "animal": ["cat", "dog", "mouse"],
}

structure = "an antropomorphic color animal"

class Modulator:
    def __init__(
        self,
        categories,
        tokenizer,
        text_encoder_1,
        text_encoder_2,
    ):
        self.category_idxs = {}
        self.categories = {}

        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2

        for category, category_values in categories.items():
            embeddings_1 = []
            embeddings_2 = []

            text_inputs = tokenizer(
                category,
                return_tensors="pt",
            )
            self.category_idxs[category] = text_inputs.input_ids[0][1]

            for word in category_values:
                text_inputs = tokenizer(
                    word,
                    return_tensors="pt",
                )

                # check if the length of the attention mask = 3 so that we are sure there is only one token.
                # if not, discard this word
                if text_inputs.attention_mask.sum() != 3:
                    print(f"removing word {word}")
                else:
                    word_id = text_inputs.input_ids[0][1]
                    print(f"token id for word {word} is {word_id}")

                    # now find the corresponding embedding for text_encoder 1
                    word_embedding_1 = text_encoder_1.text_model.embeddings.token_embedding.weight.data[word_id]
                    word_embedding_2 = text_encoder_2.text_model.embeddings.token_embedding.weight.data[word_id]

                    embeddings_1.append(word_embedding_1)
                    embeddings_2.append(word_embedding_2)
            embeddings_1 = torch.stack(embeddings_1)
            embeddings_2 = torch.stack(embeddings_2)

            self.categories[category] = {
                "embeddings_1": embeddings_1,
                "embeddings_2": embeddings_2,
            }

    
    def get_idx_embeddings(self, categorie: str, alpha: float):
        categorie_idx = self.category_idxs[categorie]

        embeddings_1 = self.categories[categorie]["embeddings_1"]
        embeddings_2 = self.categories[categorie]["embeddings_2"]

        slerped_1 = uniform_slerp(embeddings_1, alpha)
        slerped_2 = uniform_slerp(embeddings_2, alpha)

        return categorie_idx, slerped_1, slerped_2

    def set_idx_embeddings(
        self,
        categorie: str,
        alpha: float,
    ):
        categorie_idx, slerped_1, slerped_2 = self.get_idx_embeddings(
            categorie,
            alpha,
        )
        
        self.text_encoder_1.text_model.embeddings.token_embedding.weight.data[categorie_idx] = slerped_1
        self.text_encoder_2.text_model.embeddings.token_embedding.weight.data[categorie_idx] = slerped_2

if __name__=="__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        subfolder="text_encoder",
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        subfolder="text_encoder_2",
    )

    modulator = Modulator(
        categories,
        tokenizer,
        text_encoder,
        text_encoder_2,
    )

    idx_to_override, embedding_1, embedding_2 = modulator.get_idx_embeddings("color", 0.5)
    print(f"overriding {idx_to_override}")

    # override the idx embeddings
    text_encoder.text_model.embeddings.token_embedding.weight.data[idx_to_override] = embedding_1
    text_encoder_2.text_model.embeddings.token_embedding.weight.data[idx_to_override] = embedding_2