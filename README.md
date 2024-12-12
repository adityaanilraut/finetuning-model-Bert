# Sentiment Analysis with DistilBERT

## Project Overview
This project implements a sentiment analysis model using DistilBERT on the IMDB movie review dataset. The goal is to classify movie reviews as positive or negative using a fine-tuned transformer model.

## Prerequisites
- Python 3.11+
- PyTorch
- Transformers library
- Datasets library

## Installation
```bash
pip install transformers datasets torch
```

## Project Structure
```
finetuning-model-Bert/
├── Sentiment-Analysis-Final.ipynb
├── requirements.txt
└── README.md
```

## Model Training Pipeline

### 1. Dataset Preparation
- **Dataset**: IMDB Movie Reviews
- **Source**: Hugging Face Datasets Library
- **Preprocessing**:
  - Tokenization using DistilBERT tokenizer
  - Padding and truncation to 256 tokens
  - Conversion to PyTorch tensors

### 2. Model Architecture
- **Base Model**: DistilBERT (Uncased)
- **Model Type**: Sequence Classification
- **Number of Labels**: 2 (Positive/Negative)

### 3. Training Configuration
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Evaluation Strategy**: Epoch-based

## Training Script Breakdown
The training script (`Sentiment-Analysis-Final.ipynb`) consists of several key steps:

1. **Load Dataset**
   - Uses Hugging Face `load_dataset` to fetch IMDB reviews
   
2. **Preprocess Data**
   - Tokenizes text using DistilBERT tokenizer
   - Prepares data for model training
   
3. **Model Initialization**
   - Loads pre-trained DistilBERT
   - Configures for binary classification
   
4. **Training Setup**
   - Defines training arguments
   - Initializes Hugging Face Trainer
   
5. **Model Training**
   - Trains on training dataset
   - Evaluates on test dataset
   
6. **Model Saving**
   - Saves fine-tuned model and tokenizer

## Inference
The test script demonstrates prediction capabilities:
- Compares base and fine-tuned model predictions
- Displays sentiment and confidence score

### Sample Predictions
```
Base Model Predictions:
Text: This movie was absolutely fantastic!
Prediction: positive with score: 0.55

Fine-Tuned Model Predictions:
Text: This movie was absolutely fantastic!
Prediction: positive with score: 0.98
```

## Performance Metrics
- **Base Model Accuracy**: Varies
- **Fine-Tuned Model Accuracy**: Improved performance

## Potential Improvements
- Experiment with learning rates
- Try different tokenization strategies
- Increase training epochs
- Use advanced techniques like learning rate scheduling

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
MIT License

## Acknowledgements
- Hugging Face Transformers
- IMDB Dataset
