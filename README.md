# Waste Segregation Identifier

💡 **Project Idea & Problem Statement**  
Proper waste segregation is a critical global challenge, and many people are unsure how to correctly dispose of items.  
The **Waste Segregation Identifier** is a web application designed to simplify this process by providing an intuitive and immediate solution.

Our app allows users to get an instant classification of a waste item, promoting better and more sustainable disposal habits.

---

✨ **Features**  
- **Image Classification**: The app can classify an uploaded image of a waste item.  
- **Real-time Results**: Provides fast classification results in real-time on a serverless platform.  
- **Three Categories**: Classifies items into *Recyclable*, *Organic*, and *Hazardous* waste. (Removed *General Waste* category to improve accuracy and simplify the problem.)  
- **Simple Interface**: Clean, user-friendly interface powered by **Gradio**.  

---

🚀 **How It Works**  
1. **Data Preparation**: Started with thousands of images, cleaned the dataset, removed duplicates/corrupt files, and balanced classes.  
2. **Model Training**: Used **MobileNetV2 (Transfer Learning)** for higher accuracy.  
3. **App Hosting**: Model deployed on **Hugging Face Spaces** with a Python backend.  
4. **Prediction**: The backend processes the uploaded image and classifies it.  
5. **Classification**: Displays the category with the highest confidence.  

---

⚙️ **Technologies Used**  
- **Backend**: Gradio, Python, TensorFlow  
- **Model**: MobileNetV2 (Transfer Learning)  
- **Data Tools**: Pillow, Scikit-learn  
- **Deployment**: Hugging Face Spaces  

---

📊 **Live Demo**  
👉 [Try the App Here](https://huggingface.co/spaces/Krishna452002/my-waste-classifier)
