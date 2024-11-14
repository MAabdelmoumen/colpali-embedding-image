from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import uvicorn
import torch
from transformers import AutoProcessor, PaliGemmaProcessor
from PIL import Image
import io
import base64
from torch.utils.data import DataLoader
from contextlib import asynccontextmanager
from huggingface_hub import login
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import your ColPali model and ColPaliProcessor class
from colpali_engine.models import ColPali

HF_TOKEN = "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"


# Define the input schema for text embeddings
class TextsInput(BaseModel):
    texts: List[str]


# Define the input schema for image embeddings (base64-encoded images)
class ImagesInput(BaseModel):
    images: List[str]  # List of base64-encoded image strings


# Define the ColPaliProcessor class (as provided in your helper code)
class ColPaliProcessor:
    def __init__(self, model_name: str) -> None:
        self.processor: PaliGemmaProcessor = AutoProcessor.from_pretrained(model_name)

    def _process_images(self, images: List[Image.Image], max_length: int = 50):
        batch_doc = self.processor(
            text=["Describe the image."] * len(images),
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )
        return batch_doc

    def create_image_dataloader(self, images: List[Image.Image]):
        dataloader = DataLoader(
            images,  # type: ignore
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: self._process_images(x),
        )
        return dataloader

    def embed_dataloader(self, model, dataloader):
        embedding_list: List[torch.Tensor] = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            embedding_list.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return embedding_list

    def batch_embed_images(self, model, images: List[Image.Image]):
        dataloader = self.create_image_dataloader(images)
        embedding_tensors = self.embed_dataloader(model, dataloader)
        embedding_list = [tensor.tolist() for tensor in embedding_tensors]
        return embedding_list

    def _process_queries(self, queries: List[str], max_length: int = 50):
        texts_query = [
            f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            for query in queries
        ]
        batch_query = self.processor(
            images=[Image.new("RGB", (448, 448), (255, 255, 255))] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )
        del batch_query["pixel_values"]
        batch_query["input_ids"] = batch_query["input_ids"][
            ..., self.processor.image_seq_length :
        ]
        batch_query["attention_mask"] = batch_query["attention_mask"][
            ..., self.processor.image_seq_length :
        ]
        return batch_query

    def create_query_dataloader(self, queries: List[str]):
        dataloader = DataLoader(
            queries,  # type: ignore
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: self._process_queries(x),
        )
        return dataloader

    def embed_queries(self, model, dataloader):
        query_embeddings: List[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embedding_query = model(**batch_query)
            query_embeddings.extend(list(torch.unbind(embedding_query.to("cpu"))))
        return query_embeddings

    def batch_embed_queries(self, model, queries: List[str]):
        dataloader = self.create_query_dataloader(queries)
        query_tensors = self.embed_queries(model, dataloader)
        query_embeddings = [tensor.tolist() for tensor in query_tensors]
        return query_embeddings


# Define the lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    login(token=HF_TOKEN)
    model_name = "vidore/colpali"

    model = ColPali.from_pretrained(
        "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # model = ColPali.from_pretrained(
    #     "google/paligemma-3b-mix-448", torch_dtype=torch.float32, device_map="cpu"
    # ).eval()

    model.load_adapter(model_name)
    colpali_processor = ColPaliProcessor(model_name)

    # Store the model and processor in the application's state
    app.state.model = model
    app.state.processor = colpali_processor

    yield  # Control is passed to the application


# Create the FastAPI app with the lifespan event handler
app = FastAPI(lifespan=lifespan)
executor = ThreadPoolExecutor()


def decode_base64_image(image_str: str):
    """
    Decodes a base64-encoded image string into a PIL Image object.
    """
    try:
        image_data = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")


def decode_images_in_parallel(image_strings: list, max_workers: int = 4):
    """
    Decodes a list of base64-encoded image strings in parallel.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(decode_base64_image, image_strings))
    return images


@app.post("/create_embedding_for_images")
async def create_embedding_for_images(input: ImagesInput, request: Request):
    # Step 1: Decode base64-encoded images in parallel
    import time

    images = decode_images_in_parallel(input.images)

    # Step 2: Process the images with the model (replace this with your actual embedding code)
    embeddings = request.app.state.processor.batch_embed_images(
        request.app.state.model, images
    )
    return {"embeddings": embeddings}


# Endpoint to create embeddings for texts
@app.post("/create_embedding_for_texts")
async def create_embedding_for_texts(input: TextsInput, request: Request):
    texts = input.texts
    # Access the model and processor from the app's state
    embeddings = request.app.state.processor.batch_embed_queries(
        request.app.state.model, texts
    )
    return {"embeddings": embeddings}


@app.get("/")
async def health_check_():
    return {"is_ready": True}, 200


@app.get("/v1/models")
async def health_check():
    return {"is_ready": True}, 200


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
