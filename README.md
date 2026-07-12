# Fashion Recommender System

A deep learning based fashion recommendation app that suggests visually similar clothing items when you upload an image. Built with **ResNet50** for feature extraction and **k-Nearest Neighbors** for similarity search, deployed with **Streamlit**.

🔗 **Live App:** [https://kq8er8fugw5pfu6sickedw.streamlit.app/](https://kq8er8fugw5pfu6sickedw.streamlit.app/)

## How It Works

1. Upload an image of a clothing item.
2. A pretrained **ResNet50** model (with the top classification layer removed) extracts a 2048-dimensional feature vector from the image.
3. The app compares this vector against precomputed embeddings of a sample fashion product dataset using **Nearest Neighbors** (Euclidean distance).
4. The 5 most visually similar products are displayed as recommendations.

## Tech Stack

- **Python**
- **TensorFlow / Keras** — ResNet50 for feature extraction
- **scikit-learn** — Nearest Neighbors similarity search
- **Streamlit** — web app interface
- **NumPy, Pillow, OpenCV** — image processing

## Dataset

Built on a sampled subset of the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle. Feature embeddings for the sample were precomputed offline and included in the repo (`embeddings.pkl`, `filenames.pkl`) so the deployed app doesn't need to run the full dataset or re-extract features at runtime.

## Project Structure

```
├── app.py              # Offline script: extracts ResNet50 embeddings for the image dataset
├── main.py              # Streamlit app: upload image, get recommendations
├── embeddings.pkl        # Precomputed feature vectors for the sample dataset
├── filenames.pkl         # Image paths matching embeddings.pkl
├── images/               # Sample fashion product images
├── uploads/              # Stores user-uploaded query images
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/RuturajChitale/Fassion-Recommendation-System.git
cd Fassion-Recommendation-System
pip install -r requirements.txt
streamlit run main.py
```

## Future Goals

- [ ] Scale up to the full ~44k image dataset using a vector database (e.g. FAISS) instead of brute-force Nearest Neighbors
- [ ] Add category/style filters (e.g. filter recommendations by type — shirts, shoes, accessories)
- [ ] Improve recommendation quality with a model fine-tuned on fashion data instead of generic ImageNet weights
- [ ] Add a "why this was recommended" explanation (e.g. highlight similar color, pattern, or category)
- [ ] Support multi-image or text-based search (e.g. "red floral dress")
- [ ] Improve UI/UX with better layout and mobile responsiveness

## Acknowledgements

Inspired by the classic ResNet50-based fashion recommender tutorial approach, adapted here with a sampled dataset pipeline for lightweight deployment.
