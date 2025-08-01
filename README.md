# Domain Phishing Detection with Deep Learning

🛡️ An AI-powered cybersecurity solution for detecting phishing URLs using Convolutional Neural Networks and LSTM architectures.

## 🎯 Project Overview

This project implements advanced deep learning techniques to identify malicious phishing URLs through character-level analysis. By treating URLs as sequences of characters, our CNN-LSTM hybrid model achieves **98.77% accuracy** in distinguishing between legitimate and phishing websites.

## 🚀 Key Features

- **Character-Level Analysis**: Advanced tokenization of URLs up to 256 characters
- **Hybrid CNN-LSTM Architecture**: Combines pattern detection with sequential modeling
- **High Performance**: Achieves 98.77% accuracy with 95.48% precision
- **Real-time Detection**: Optimized for fast URL classification
- **Comprehensive Pipeline**: Complete data preprocessing to model deployment

## 📊 Dataset

- **Primary Dataset**: 549,346 labeled URLs (phishing vs legitimate)
- **Preprocessing**: Character-level tokenization with one-hot encoding
- **Feature Engineering**: Domain analysis, SSL status, character frequency
- **Data Shape**: 3D tensors (samples, 68, 256) for neural network input

## 🏗️ Architecture

### Model Components
- **Embedding Layer**: Maps characters to dense vectors (68 dimensions)
- **2D Convolutional Layer**: Detects local patterns in URL structure
- **MaxPooling Layer**: Reduces dimensionality while preserving features
- **LSTM Layer**: Captures sequential dependencies
- **Dense Output**: Binary classification with sigmoid activation

### Performance Metrics
- **Accuracy**: 98.77%
- **Precision**: 95.48%
- **Recall**: 98.45%
- **Model Parameters**: ~1.2M trainable parameters

## 📁 Project Structure

```
domain-phishing/
├── 1D-CNN.ipynb                    # Primary CNN implementation
├── CNN-LSTM Classifier.ipynb       # Hybrid CNN-LSTM model
├── LargeDataCNN.ipynb              # Large-scale data processing
├── Preprocess_Layer.ipynb          # Data preprocessing pipeline
├── data/
│   ├── phishing_site_urls.csv      # Raw dataset (549K URLs)
│   ├── conv_7997_68_256.csv        # Preprocessed tensor data
│   └── pre_conv_df_7997.csv        # Processed features
├── models/
│   └── model.h5                    # Trained model weights
├── AI-in-Cyber.md                  # Comprehensive research slides
├── PROJECT_DESCRIPTION.md          # Detailed project documentation
└── requirements.txt                # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x (GPU recommended)
- Minimum 8GB RAM

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
```bash
pip install tensorflow keras
```

## 🚀 Quick Start

### 1. Data Preprocessing
```bash
jupyter notebook Preprocess_Layer.ipynb
```

### 2. Train the Model
```bash
jupyter notebook CNN-LSTM\ Classifier.ipynb
```

### 3. Evaluate Performance
```bash
jupyter notebook 1D-CNN.ipynb
```

## 📈 Usage Examples

### Basic URL Classification
```python
# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('models/model.h5')

# Preprocess URL
url = "suspicious-website.com/login"
processed_url = preprocess_url(url)  # Your preprocessing function

# Predict
prediction = model.predict(processed_url)
is_phishing = prediction[0][0] > 0.5
```

### Batch Processing
```python
# Process multiple URLs
urls = ["example.com", "phishing-site.net", "legitimate-bank.com"]
predictions = batch_predict(model, urls)
```

## 🔬 Research Context

This project is part of research on **"AI in Cybersecurity: Detecting Domain Phishing"** and demonstrates:

- **Proactive vs Reactive Security**: Moving beyond traditional rule-based detection
- **Character-Level Deep Learning**: Novel approach to URL analysis
- **Hybrid Architecture**: Combining CNN feature extraction with LSTM modeling
- **Practical AI Application**: Real-world cybersecurity implementation

## 📚 Documentation

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)**: Comprehensive technical documentation
- **[AI-in-Cyber.md](AI-in-Cyber.md)**: Research presentation and theoretical background
- **Jupyter Notebooks**: Step-by-step implementation with detailed explanations

## 🎯 Applications

### Enterprise Security
- Integration with existing cybersecurity frameworks
- Real-time URL validation in corporate environments
- Email security and link scanning

### Consumer Protection
- Browser extensions for real-time protection
- Mobile app integration
- Social media link verification

### Research & Education
- Cybersecurity training datasets
- Deep learning curriculum examples
- AI ethics and security research

## 🔮 Future Enhancements

- **Multi-modal Analysis**: Incorporate website content and visual elements
- **Real-time API**: Deploy as scalable web service
- **Mobile Optimization**: Lightweight models for mobile devices
- **Adversarial Robustness**: Defense against sophisticated evasion techniques

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Author**: Dominik Huffield
- **Research Focus**: AI in Cybersecurity
- **Academic References**: CNN-based malicious URL detection papers
- **Dataset Sources**: Community-contributed phishing URL datasets

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Lead**: Dominik Huffield
- **Research Area**: AI in Cybersecurity
- **Focus**: Deep Learning for Threat Detection

---

⭐ **Star this repository** if you find it helpful for your cybersecurity or machine learning projects!

🛡️ **Stay Safe Online** - Use AI to fight fire with fire against cyber threats.
