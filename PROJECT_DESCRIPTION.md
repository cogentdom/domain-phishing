# Domain Phishing Detection Project

## Overview

This project implements a deep learning-based approach to detect phishing URLs using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The system analyzes URL characteristics at the character level to classify URLs as malicious (phishing) or legitimate.

## Purpose

The primary goal of this project is to develop an AI-powered cybersecurity solution that can:

- **Proactively detect phishing attempts** before users interact with malicious websites
- **Analyze URL patterns** using deep learning to identify sophisticated phishing techniques
- **Provide automated threat detection** that adapts to evolving phishing strategies
- **Demonstrate the application of AI in cybersecurity** for educational and research purposes

## Project Background

This work is part of a broader research initiative titled "AI in Cybersecurity: A Deep Dive into Detecting Domain Phishing" by Dominik Huffield. The project addresses the critical need for advanced phishing detection methods as traditional rule-based approaches struggle to keep pace with increasingly sophisticated cyber threats.

### Key Motivations:
- **Reactive vs Proactive Security**: Moving from traditional reactive cybersecurity measures to AI-driven proactive threat detection
- **Scale and Automation**: Enabling network-wide behavior monitoring and automated flagging of malicious actors
- **Adaptability**: Creating systems that can learn and adapt to new phishing techniques

## Technical Architecture

### Data Processing Pipeline

1. **Dataset**: 
   - Primary dataset: `phishing_site_urls.csv` (~549,346 URLs)
   - Preprocessed datasets: `conv_7997_68_256.csv` and `pre_conv_df_7997.csv`

2. **URL Preprocessing**:
   - Character-level tokenization of URLs (max length: 256 characters)
   - One-hot encoding using ASCII character set (characters 33-127)
   - Conversion to 3D tensors with shape `(samples, 68, 256)`
   - Binary classification: `1` for malicious URLs, `0` for legitimate URLs

3. **Feature Engineering**:
   - Domain and subdomain analysis
   - URL length and path length
   - SSL certification status
   - Character frequency analysis (dots, dashes, underscores)
   - DNS server and IP address extraction

### Model Architecture

The project implements multiple deep learning approaches:

#### CNN-LSTM Hybrid Model
- **Embedding Layer**: Maps character indices to dense vectors (embedding dimension: 68)
- **2D Convolutional Layer**: Detects local patterns in URL character sequences
- **MaxPooling Layer**: Reduces dimensionality while preserving important features
- **LSTM Layer**: Captures sequential dependencies in URL structure
- **Dense Output Layer**: Binary classification with sigmoid activation

#### Model Specifications:
- Input shape: `(68, 256)` representing character sequences
- Total parameters: ~1.2M trainable parameters
- Optimization: Adam optimizer with binary cross-entropy loss
- Metrics: Precision, Recall, and Accuracy

## Implementation Files

### Jupyter Notebooks
- **`1D-CNN.ipynb`**: Primary CNN implementation for URL classification
- **`CNN-LSTM Classifier.ipynb`**: Hybrid CNN-LSTM model with advanced architecture
- **`LargeDataCNN.ipynb`**: Scaled implementation for large dataset processing
- **`Preprocess_Layer.ipynb`**: Data preprocessing and feature engineering pipeline

### Data Files
- **`data/phishing_site_urls.csv`**: Raw dataset of labeled URLs
- **`data/conv_7997_68_256.csv`**: Preprocessed tensor data for model training
- **`models/model.h5`**: Trained model weights

### Documentation
- **`AI-in-Cyber.md`**: Comprehensive presentation slides covering theoretical background
- **`outline.txt`**: Project structure and research outline
- **Research Papers**: Supporting academic literature on CNN-based phishing detection

## Performance Metrics

The implemented models achieve:
- **Accuracy**: ~98.77%
- **High Precision**: ~95.48%
- **High Recall**: ~98.45%

These metrics demonstrate the effectiveness of character-level CNN analysis for phishing detection.

## Technical Requirements

### Dependencies
```
numpy
matplotlib
pandas
seaborn
kaggle
redis
scikit-learn
notebook
tensorflow/keras
```

### System Requirements
- Python 3.7+
- TensorFlow 2.x with GPU support (recommended)
- Minimum 8GB RAM for model training
- Storage: ~1GB for datasets and models

## Research Context

This project contributes to the broader field of AI-driven cybersecurity by:

1. **Advancing Character-Level Analysis**: Demonstrating the effectiveness of treating URLs as character sequences for deep learning
2. **Hybrid Architecture Exploration**: Combining CNN feature extraction with LSTM temporal modeling
3. **Practical Implementation**: Providing a complete pipeline from data preprocessing to model deployment
4. **Educational Value**: Serving as a comprehensive example of applying deep learning to cybersecurity challenges

## Applications and Impact

### Immediate Applications:
- **Enterprise Security**: Integration with existing cybersecurity frameworks
- **Browser Extensions**: Real-time URL validation before page navigation
- **Email Security**: Scanning links in email communications
- **API Services**: Providing phishing detection as a cloud service

### Broader Impact:
- **Proactive Threat Detection**: Shifting from reactive to predictive cybersecurity
- **Automated Defense**: Reducing reliance on manual threat analysis
- **Adaptive Learning**: Systems that improve with exposure to new threats

## Future Directions

### Technical Enhancements:
- **Multi-modal Analysis**: Incorporating website content, visual elements, and metadata
- **Transfer Learning**: Adapting models to detect other types of malicious URLs
- **Real-time Processing**: Optimizing models for low-latency deployment
- **Federated Learning**: Enabling privacy-preserving collaborative learning

### Research Opportunities:
- **Adversarial Robustness**: Testing model resilience against sophisticated evasion techniques
- **Cross-lingual Support**: Extending detection to international domain names
- **Temporal Analysis**: Understanding how phishing techniques evolve over time

## Educational Value

This project serves as an excellent educational resource for:
- **Data Science Students**: Learning practical deep learning implementation
- **Cybersecurity Professionals**: Understanding AI applications in threat detection
- **Researchers**: Exploring novel approaches to malicious URL detection
- **Industry Practitioners**: Implementing AI-driven security solutions

## Conclusion

The Domain Phishing Detection Project demonstrates the powerful potential of deep learning in cybersecurity applications. By treating URL analysis as a character-level sequence classification problem, the project achieves high accuracy while providing insights into the application of CNNs and LSTMs for cybersecurity tasks.

The comprehensive implementation, from data preprocessing to model evaluation, provides a complete framework for understanding and deploying AI-driven phishing detection systems. This work contributes to the growing field of AI-powered cybersecurity and offers practical solutions for protecting users and organizations from increasingly sophisticated phishing attacks.

---

*Author: Dominik Huffield*  
*Project Focus: AI in Cybersecurity - Deep Learning for Phishing Detection*