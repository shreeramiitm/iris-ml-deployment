# 🌸 Iris Flower Classification API

A Machine Learning deployment project that predicts the species of an Iris flower using a **K-Nearest Neighbors (KNN)** model.

The trained model is served through a **FastAPI REST API**, containerized using **Docker**, and deployed using **Render**.

---

## 🚀 Project Workflow

```text
Iris Dataset
     ↓
KNN Model
     ↓
model.pkl
     ↓
FastAPI
     ↓
Docker
     ↓
GitHub
     ↓
Render
     ↓
Live API
```

---

## 🧠 Model

The project uses the Iris dataset with four input features:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

### Classes

| Class | Species |
| :---: | :--- |
| `0` | Setosa |
| `1` | Versicolor |
| `2` | Virginica |

### Algorithm
- **K-Nearest Neighbors (KNN)**

The trained model is saved as:
```text
model/model.pkl
```

---

## ⚙️ Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, NumPy, KNN
- **API Framework:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Version Control & CI/CD:** Git, GitHub
- **Cloud Platform:** Render
- **Frontend Interface:** HTML / JavaScript

---

## 📁 Project Structure

```text
iris-ml-deployment/
│
├── app/
│   └── main.py
│
├── model/
│   └── model.pkl
│
├── static/
│   └── ...
│
├── ML_Worshop_iris_Classification.ipynb
├── Dockerfile
├── requirements.txt
├── index.html
└── README.md
```

---

## 🔌 API Endpoints

### 🏠 Home
- **Endpoint:** `GET /`
- **Description:** Returns the Iris classification interactive web interface.

### ❤️ Health Check
- **Endpoint:** `GET /health`
- **Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 🔮 Prediction
- **Endpoint:** `POST /predict`

#### Request Body
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Response Body
```json
{
  "predicted_class": 0,
  "species": "setosa"
}
```

---

## 📖 API Documentation

FastAPI automatically provides interactive API documentation (Swagger UI).

- **Local:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Production:** `https://YOUR-RENDER-URL.onrender.com/docs`

---

## 🐳 Docker

The application is fully containerized using Docker for consistent local development and deployment.

### 1. Build the Docker Image
```bash
docker build -t iris-ml-api .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 iris-ml-api
```

The API will be available locally at [http://localhost:8000](http://localhost:8000).

---

## ☁️ Deployment

The project is deployed on **Render** using Docker.

```text
GitHub Repository
       ↓
     Render
       ↓
  Docker Build
       ↓
Container Deployment
       ↓
Live FastAPI Application
```

### 🌐 Live Links
- **Live Application:** https://iris-ml-deployment-odub.onrender.com/
- **Interactive Documentation:** https://iris-ml-deployment-odub.onrender.com/docs

---

## 📸 Screenshots
<img width="1280" height="680" alt="image" src="https://github.com/user-attachments/assets/0b3400b3-997e-4d30-8ae4-a4ed38b9c1f8" />
_ _ _ _ _ _ _ _ _ 
<img width="970" height="631" alt="image" src="https://github.com/user-attachments/assets/4405536f-bd87-46cd-9d7f-4397f828e4b3" />


---

## 🧪 Example Prediction

### Input
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Output
```json
{
  "predicted_class": 0,
  "species": "setosa"
}
```

---

## 🎯 Learning Outcomes

This project demonstrates the complete end-to-end Machine Learning lifecycle:
- Training and validating a machine learning model on tabular data.
- Serializing models into portable `.pkl` artifacts.
- Designing schema-validated REST APIs using FastAPI and Pydantic.
- Containerizing applications using reproducible Docker environments.
- Managing code and continuous deployment triggers using GitHub.
- Deploying containerized ML services to Render.
- Inspecting real-time production health checks and request logs.

---

## 🛠️ Project Workflow Summary

```text
        ┌─────────────────┐
        │   Iris Dataset  │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │   KNN Training  │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │    model.pkl    │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │     FastAPI     │
        │    /predict     │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │      Docker     │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │     GitHub      │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │     Render      │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │   Live API      │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │ Logging &       │
        │ Monitoring      │
        └─────────────────┘
```

---

## 👨‍💻 Author

**Shreeram Hirani** 
Electronics & Communication Engineering  
*Birla Vishvakarma Mahavidyalaya (BVM)*

---

## 🎓 Workshop

**Getting Started with ML in Production**  
*Practical deployment pipeline: Model → FastAPI → Docker → GitHub → Render → Logging & Monitoring*

⭐ **Iris Flower Classification API — End-to-End Machine Learning Deployment**
