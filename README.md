# Sentiment Analysis App

![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> A clean, production-minded Streamlit app that reads text and returns generic & finance-aware sentiment labels and confidence scores.

---

## Why this project

Market language is different: the same phrase can mean different things in finance compared to casual chat. This app is designed to:
- prioritize clarity (human-readable outputs),
- be resilient in deployment (tips to avoid first-run timeouts),
- and be easy to extend (swap the model, add dashboards).

---

## Quick visual 
![Demo](assets/demo.gif)  

---

## Project structure


- **app.py** - main Streamlit application script  
- **requirements.txt** - Python dependencies  
- **setup.sh** - shell script for environment setup  
- **runtime.txt** - runtime configuration (for deployment, e.g. Heroku)  
- **.devcontainer/** - config for VSCode / dev container setup  
- **Data/** - datasets, corpora, lexicons etc.  
- **finance/** - finance-specific modules, models, tools  
- **notebooks/**- Jupyter notebooks used during experimentation / prototyping  

---

## Installation & Setup

Below is a typical setup for development and running locally.

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ani-404/Sentiment-Analysis-App.git
   cd Sentiment-Analysis-App
   ```
2. **(Optional) Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application locally**
   ```bash
   streamlit run app.py
   ```
---


   


