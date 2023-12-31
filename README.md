# **Project Report: Image-Text Multi-Modal Deep Learning**

#### CLIP Model - Implementation from Scratch - ResNet50 / ViT - BERT : https://www.kaggle.com/code/shirshmall/clip-model-implementation-resnet-vit-distil-bert/notebook
#### Fine Tuned CLIP Model: https://huggingface.co/shirsh10mall/Fine_Tuned_CLIP_Model
#### Image Captioning Model - Implementation from Scratch - GRU - Attention Layers: https://www.kaggle.com/code/shirshmall/clip-model-implementation-resnet-vit-distil-bert/notebook
#### Fine Tuned Image Captioning (ViT-BERT) Model - https://huggingface.co/shirsh10mall/Image_Captioning_FineTune_ViT_BERT_GCC
#### Google's Conceptual Captioning Dataset with Embeddings: https://huggingface.co/datasets/shirsh10mall/Image_Captioning_GCC_Embeddings


### *Abstract:*
In this report, I present an in-depth exploration of an Image-Text Multi-Modal Deep Learning project aimed at achieving two primary objectives: Image Captioning and Image Retrieval. The project leverages a trifecta of diverse datasets, employs advanced deep learning models, and culminates in a user-friendly web application. Despite encountering challenges, the project demonstrates successful implementation and noteworthy outcomes.

### 1. **Dataset Collection and Preprocessing:**
Three datasets are harnessed to encapsulate the multi-modal landscape: Google's Conceptual Captioning dataset, the Flickr 8k Dataset, and a custom dataset generated from ChatGPT and Google Images using Selenium. Each dataset undergoes meticulous preprocessing and labeling, ensuring their suitability for model training and evaluation.

### 2. **Image Captioning:**
Two distinct models are pursued for Image Captioning. The first model comprises an Encoder-Decoder architecture incorporating a GRU mechanism and Attention layers. The image encoder utilizes pre-trained MobileNetV2 or ResNet50 models, and the decoder is fashioned with embeddings, GRU, and Attention layers. This model's initial implementation yielded suboptimal results due to issues of repeated and erroneous caption generation. Subsequently, a ViT BERT pre-trained model, harnessed through Hugging Face's PyTorch setup, exhibited superior performance. Fine-tuned across the combined dataset, the ViT BERT model showcases promising results in both training and testing phases, evaluated through the Rouge score metric.

### 3. **Image Retrieval:**
In pursuit of refining the Image Retrieval component, I undertook the implementation of both a Vision Transformer and a CLIP model from the ground up, utilizing the PyTorch framework. I accomplished this by leveraging a pre-trained DistilBERT model, which served as the backbone for the CLIP architecture. This strategic decision resulted in robust training outcomes, enhancing the overall efficiency of the image retrieval system.

Although initial attempts at constructing a CLIP model faced challenges in parameter updates, a strategic shift was made to fine-tune the pre-trained ViT Clip model provided by OpenAI. This decision yielded promising results across all three datasets. Subsequently, the fine-tuned CLIP model played a pivotal role in generating insightful image embeddings.

To optimize the image retrieval process, these embeddings were meticulously organized through the implementation of FAISS indexing. This facilitated a streamlined and efficient similarity-based search mechanism, contributing to the overall effectiveness of the system.

### 4. **Model Deployment:**
A pivotal culmination of the project is the development of an interactive web application using FastAPI, HTML, and CSS. The application is partitioned into two user-friendly tabs: one for image captioning and another for image retrieval. In the former, users upload images to generate captions using the ViT BERT model. In the latter, users input text queries to retrieve images based on text-image similarity scores.

### *Challenges Faced:*
The project was not devoid of challenges. Initial attempts to build a CLIP model from scratch faced obstacles in terms of parameter updates, hindering efficient training. However, this was mitigated through the adoption of a pre-trained ViT Clip model.

*Conclusion:*
In conclusion, the Image-Text Multi-Modal Deep Learning project successfully tackles the intricate synergy between images and text. Through diverse datasets, advanced models, and meticulous deployment, the project achieves its dual objectives, showcasing promising outcomes in both image captioning and retrieval.

This project report stands as a testament to the potential of deep learning in unravelling the intricate tapestry of multi-modal interactions, with practical applications across domains. It serves as an ideal portfolio piece, exemplifying the ability to harness deep learning techniques for real-world problem-solving.


### Web App Sample:

1. Image Captioning:
   
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/1168c15f-08b7-4d96-a506-514eb58a19ff)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/cde0908c-40c5-4077-aa7c-45b41efa768b)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/cc75488c-814e-4c56-9634-de39c82bb816)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/b0ae826e-16df-4a4c-875d-75dd9af7c7a7)



2. Image Retrieval:

![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/edc600b1-435b-437c-aa4e-024992ac6fe4)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/e2a17f31-8159-4d55-a847-7344be636bdc)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/a8e61c17-2c1e-4e1f-9cc7-b348c0190f50)
![image](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/6314bc4d-17af-4d2c-b268-e51dd41975f7)


3. Docker Terminal:
![Screenshot_13](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/df60de6e-8ace-4b82-8629-961481d4b33b)
![Screenshot_1](https://github.com/shirsh10mall/Image-Text-Multi-Models/assets/87264071/549bb387-a9b7-4f01-bed0-81b5a21f9b66)

