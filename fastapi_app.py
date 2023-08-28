# Description: FastAPI application for the Image Retrieval and Image Captioning tasks.

from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from PIL import Image
import io
# Image Retrieval libraries
from datasets import Dataset, load_from_disk
import os
from PIL import Image
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import torch
import os
current_directory = os.getcwd()
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from torchvision import transforms
import uvicorn

# Load pre-trained CLIP model and tokenizer
base_model_name =  "openai/clip-vit-base-patch16" # "shirsh10mall/Fine_Tuned_CLIP_Model" # openai/clip-vit-base-patch16 "openai/clip-vit-large-patch14" 
clip_model = CLIPModel.from_pretrained( os.path.join( current_directory, "Fine_Tuned_CLIP_Model"))
clip_tokenizer = CLIPProcessor.from_pretrained(base_model_name)
dataset = load_from_disk( os.path.join(current_directory, "./Image_Captioning_GCC_Embeddings_1k"))
dataset = dataset["train"]
dataset.load_faiss_index('image_embeddings', 'image_embeddings.faiss')

def image_retrieval_sequentially(query, counter_n):
    # Encode the query text
    query_input = clip_tokenizer(query, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    query_embedding = clip_model.get_text_features(**query_input)

    # Perform image retrieval
    scores, retrieved_examples = dataset.get_nearest_examples('image_embeddings', query_embedding[0].detach().numpy(), k=counter_n)
    return retrieved_examples


#  Image Captioning libraries
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from torchvision import transforms
decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_special_tokens=True)
if decoder_tokenizer.pad_token is None:
    decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = VisionEncoderDecoderModel.from_pretrained(os.path.join(current_directory, "Image Captioning-Fine-Tune ViT-BERT Model Flickr8k/Image_Captioning_Fine_Tune_ViT_BERT_model_flickr8k"))
transform = transforms.Compose([ transforms.Resize((224,224)), transforms.ToTensor() ])

def create_caption(image):
    image = transform( image )
    channels = image.shape[0]
    if channels==1:
        image = torch.stack( [image]*3 , dim=1)
    elif channels==4:
        image = image[:3, :, :]
    image = image.unsqueeze(0)
    predictions = decoder_tokenizer.decode(model.generate(pixel_values=image)[0], skip_special_tokens=True) # 
    caption = predictions.replace(".","") 
    return caption

app = FastAPI()

# Mount a directory for static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Image Captioning route
@app.post("/image_captioning/")
async def image_captioning(request: Request, file: UploadFile = File(...)):

    # Process the uploaded image here
    image_bytes = await file.read()
    uploaded_image_path = current_directory+"\\static\\uploaded\\"+file.filename
    uploaded_images = []  # Store uploaded image paths
    with open(uploaded_image_path, "wb") as f:
        f.write(image_bytes)
    uploaded_images.append(uploaded_image_path)

    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform image captioning using your preferred fine tuned model
    generated_caption = create_caption(image)
    return templates.TemplateResponse("captioning_content.html", { "request": request, "caption": generated_caption, "uploaded_image": "uploaded/" + file.filename})

# Image Retrieval route
@app.post("/image_retrieval/")
async def image_retrieval(request: Request, query_text: str = Form(...), num_images: int = Form(...)):
    # Perform image retrieval based on the query_text and num_images
    retrieved_examples = image_retrieval_sequentially(query_text, counter_n=num_images)
    # Retrieve relevant images using your preferred method/model
    # Save the retrieved images to the static folder
    retrieved_images = []
    for i, image in enumerate(retrieved_examples["image_data"]):
        image_name = "retrieved_image_"+str(i+1)+".jpg"
        image.save(os.path.join("static", image_name))
        retrieved_images.append(image_name)
    
    # retrieved_images = [ "/"+image_name for image_name in retrieved_images]
    return templates.TemplateResponse("retrieval_content.html", {"request": request, "retrieved_images": retrieved_images})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
