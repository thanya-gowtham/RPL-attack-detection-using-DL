# 🚀 RPL Intrusion Detection System

A high-accuracy (99%+) deep learning–based intrusion detection system that detects five types of network attacks in RPL-based IoT environments using a hybrid CNN-GRU architecture.

## 📁 Project Structure

```
rpl-intrusion-detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                          # Data directory
│   ├── raw/                       # Raw dataset files
│   │   ├── blackhole.csv
│   │   ├── dodag.csv
│   │   ├── flooding.csv
│   │   └── rank.csv
│   ├── processed/                 # Processed data files
│   │   └── label_encoder_classes.npy
│   ├── 5class_dataset.csv         # Balanced 5-class dataset
│   └── 5class_test_input.csv      # Test input data
├── models/                        # Trained models
│   ├── ultimate_model.pth         # Best trained model (99%+ accuracy)
│   └── scaler.joblib             # Feature scaler
├── src/                          # Source code
│   ├── models/
│   │   └── cnn_gru.py            # CNN-GRU model architecture
│   └── __init__.py
├── data_processing/               # Data processing scripts
│   └── create_5class_dataset.py  # Create balanced dataset
├── training/                      # Training scripts
│   └── train_ultimate_model.py   # Train the ultimate model
├── notebooks/                     # Jupyter notebooks
├── reports/                       # Reports and visualizations
└── tests/                         # Test files
```

# 🚀 RPL Intrusion Detection System (99%+ Accuracy)

## 🔥 Key Highlights
- ✅ **99.5% overall accuracy** across 5 attack classes  
- ✅ Detects **Blackhole, DODAG, Flooding, Rank, and Normal traffic**  
- ✅ Hybrid **CNN + GRU** model for spatial and temporal feature learning  
- ✅ Real-time predictions via **Streamlit web application**  
- ✅ Supports both **batch CSV predictions** and **single-instance inference**

## 🧠 Tech Stack
- **Deep Learning:** PyTorch  
- **Frontend:** Streamlit  
- **Data Processing:** NumPy, Pandas, Scikit-learn  
- **Model Architecture:** CNN-GRU Hybrid Network  

## ⚙️ Model Overview
| Component | Details |
|--------|------------|
| Input Features | 14 network traffic parameters |
| Architecture | 4-layer CNN-GRU |
| Hidden Units | 1024 |
| Output | 5-class classification |
| Accuracy | **99.5%** |

## 🚀 Run Locally

```bash
git clone https://github.com/achugowda/rpl-intrusion-detection.git
cd rpl-intrusion-detection
pip install -r requirements.txt
streamlit run app.py
```
## 🎉 Results

The system successfully detects all 5 attack types with 99%+ accuracy, providing reliable intrusion detection for RPL networks.
