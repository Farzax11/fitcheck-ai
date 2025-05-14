# 👗 FitCheck AI – Outfit Rating App

FitCheck AI is a simple AI-powered web application that allows users to upload an image of an outfit and receive a fashion score along with its predicted category (e.g., Casual, Formal, etc.).

This version is a **basic prototype** to demonstrate the core functionality of the system using a trained model.

---

## 🚀 Features

- Upload an image of an outfit.
- View the predicted **style category**.
- Get a **fashion score** (out of 10).
- Clean and simple Streamlit-based interface.

---

## 🧠 How It Works

- A **pre-trained ResNet18** model was fine-tuned on a fashion dataset.
- The model predicts the outfit category and assigns a score.
- The app uses **Streamlit** for the frontend and Python for model inference.

---

## 📁 Dataset Note

> **Note:**  
> Due to GitHub file size limitations, the dataset (originally containing 44,000+ images) is **not included in this repository**.  
> The current version was trained using a smaller subset of the dataset (1,000 samples) for demo purposes.

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fitcheck-ai.git
cd fitcheck-ai

Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt

Run the app:
streamlit run app.py

⚙️ Project Structure
fitcheck-ai/
├── app.py                 # Streamlit app
├── model.py               # Model prediction logic
├── train_model.py         # Model training script
├── fashion_model.pth      # Trained model (not uploaded to GitHub due to size)
├── styles.csv             # Metadata CSV file
├── images/                # Outfit images folder (not uploaded to GitHub)
├── requirements.txt       # Python dependencies
└── README.md              # Project description

 Future Development
Improve accuracy with a larger and balanced dataset.

Add more outfit labels and subcategories.

Allow users to receive style suggestions or improvements.

Optimize the model and deploy on platforms like Hugging Face Spaces or Heroku.


🙋‍♀️ Author
Created as a mini AI project by FATHIMA FARZANA.
Part of an EDUNET FOUNDATION's capstone project submission.
