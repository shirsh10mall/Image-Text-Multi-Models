# Image-Text-Multi-Models


Fine Tuned CLIP Model: https://huggingface.co/shirsh10mall/Fine_Tuned_CLIP_Model

Fine Tuned Image Captioning (ViT-BERT) Model - https://huggingface.co/shirsh10mall/Image_Captioning_FineTune_ViT_BERT_GCC

Google's Conceptual Captioning Dataset with Embeddings: https://huggingface.co/datasets/shirsh10mall/Image_Captioning_GCC_Embeddings


**Harnessing the Power of Multi-Modal Intelligence: A Deep Dive into Image-Text Multi-Model Project**

In the ever-evolving landscape of AI, the fusion of image and text analysis has sparked a new era of multi-modal intelligence. This project embarks on an exhilarating journey to master two significant objectives: Image Captioning and Image Retrieval, where images and text interact harmoniously to enrich understanding and user experiences.

**A Trio of Datasets Unveil Diversity and Complexity:**

Three diverse datasets lay the groundwork for this ambitious project's voyage:

1. **Conceptual Captioning Dataset**: A fraction of Google's vast repository, this collection boasts 145k images with associated captions harvested from alt-text HTML attributes. Each image encapsulates a distinct theme, and the captions are culled from the web, offering an array of styles and expressions.

2. **Flickr 8k Dataset**: Purposefully designed for benchmarking image description and retrieval tasks, this collection features 8,000 images, each paired with five distinct captions. The dataset's careful curation ensures representation across various scenes and contexts.

3. **Custom Dataset Creation**: Fusing creativity with technology, a bespoke dataset emerges, composed of 1,000 image-caption pairs. The sentences, meticulously generated using ChatGPT, span an array of activities involving living and non-living entities. Leveraging Selenium's web scraping capabilities, high-resolution images from Google enrich this unique dataset, fostering a blend of innovation and practicality.

**Embarking on the Image Captioning Odyssey:**

The project's first objective, image captioning, materializes through two distinct models:

1. **Encoder-Decoder Model with Attention**: Combining the prowess of image encoders (MobileNetV2 or ResNet50) with decoder layers (Embedding, GRU, and Attention), this model synthesizes contextual captions. Images are encoded, producing latent features that serve as the context for the text decoder, culminating in coherent and expressive captions. TensorFlow orchestrates this journey, yet initial attempts yielded less-than-desirable results, plagued by repetitive and inaccurate captions.

2. **ViT BERT Pre-trained Model**: Pivoting towards the ViT BERT model, leveraging PyTorch via hugging face's setup, marks a transformative stride. Fine-tuning on a merged dataset propels this model to exceptional heights, delivering captions that seamlessly capture image nuances. The resulting efficacy, measured through the lens of Rouge score metrics, underscores the power of this approach.

**Charting the Complex Terrain of Image Retrieval:**

Navigating the intricate realm of image retrieval involves traversing two pathways:

1. **From-Scratch CLIP Model**: The inception of a Custom Learning Image Pretext (CLIP) model unveils TensorFlow's prowess. This model endeavours to correlate textual and visual content. Yet, roadblocks emerged as training dynamics posed challenges, hampering parameter updates and progress.

2. **Fine-Tuned ViT Clip Pre-trained Model**: Pivoting towards leveraging a pre-trained model (ViT Clip) and fine-tuning proved instrumental. Through meticulous training on a hybrid dataset, performance milestones were achieved. The extraction of image embeddings and their organization via FAISS indexing streamlined retrieval, propelling efficient search.

**Culmination in a User-Friendly Interface:**

The project's pinnacle is realized through an interactive web application, melding the realms of technology and user experience. FastAPI, HTML, and CSS converge to create an accessible platform, fostering interaction with the project's potent capabilities.

1. **Image Captioning Tab**: Users can seamlessly upload images, where models generate descriptive captions that reflect the intricate details within the images.

2. **Image Retrieval Tab**: A novel facet unveils itself, enabling users to input textual queries and specify the desired number of images. Here, text embeddings guide image selection, creating a symbiotic relationship between queries and curated visual results.



With this project, the intricate dance between images and text results in a harmonious fusion of technology and creativity, enhancing user experiences in the world of visual and semantic exploration.
