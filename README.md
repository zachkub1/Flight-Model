# ✈️ Flight Price Predictor

A machine learning web application that predicts flight prices based on various factors like airline, route, timing, and class of travel.

## 🚀 Quick Start

### 1. Train the Model
```bash
python main.py
```
This will:
- Load and preprocess the flight data
- Train a Random Forest model
- Perform hyperparameter tuning
- Save the best model as `flight_model.pkl`

### 2. Run the Web App
```bash
streamlit run app.py
```
Open your browser to the provided URL (usually `http://localhost:8501`)

## 📋 Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas scikit-learn scipy numpy matplotlib
```

## 🌐 Deployment Options

### Option 1: Local Development
- Run `streamlit run app.py` on your machine
- Access via `http://localhost:8501`

### Option 2: Streamlit Cloud (Free)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### Option 3: Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

### Option 4: Railway
1. Connect your GitHub repo to [Railway](https://railway.app)
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py --server.port=$PORT`

### Option 5: Google Cloud Run
1. Create a Dockerfile
2. Build and deploy to Cloud Run
3. Set environment variables for port

## 🏗️ Project Structure

```
Flight/
├── main.py              # Model training script
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── Clean_Dataset.csv   # Flight data (not included)
└── flight_model.pkl    # Trained model (generated after training)
```

## 🔧 Features

- **Interactive UI**: User-friendly web interface
- **Real-time Predictions**: Instant price estimates
- **Multiple Airlines**: Support for major Indian airlines
- **Route Selection**: Choose from/to major Indian cities
- **Time Preferences**: Select departure/arrival times
- **Class Options**: Economy vs Business class
- **Performance Metrics**: Model accuracy indicators

## 📊 Model Performance

The Random Forest model provides:
- High R² score for accurate predictions
- Low MAE for reliable estimates
- Feature importance analysis
- Cross-validation optimization

## 🚨 Troubleshooting

### Common Issues:

1. **Model not found error**
   - Ensure you've run `main.py` first
   - Check that `flight_model.pkl` exists

2. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Port already in use**
   - Use different port: `streamlit run app.py --server.port 8502`

4. **Memory issues**
   - Reduce `n_estimators` in the model
   - Use smaller dataset for training

## 📈 Future Enhancements

- [ ] Add more airlines and routes
- [ ] Include seasonal pricing factors
- [ ] Real-time data integration
- [ ] User authentication
- [ ] Prediction history
- [ ] API endpoints for mobile apps

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License. 