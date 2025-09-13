import gradio as gr #type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
from PIL import Image
import json

model = load_model("best_model.h5")

with open("labels.json", "r") as f:
    labels = json.load(f)
label_list = [labels[str(i)] for i in range(len(labels))]

def predict_image(img):
    try:
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = label_list[predicted_index]
        confidence = predictions[predicted_index] * 100

        css_class = f"result-{predicted_label.lower()}"
        html_output = f"""
            <div id='result-container'>
                <p id="result-text" class="{css_class}">
                    This is <b>{predicted_label}</b>. (Confidence: {confidence:.2f}%)
                </p>
            </div>
        """
        return html_output
    except Exception as e:
        return f"<span style='color:red;'>Error: {str(e)}</span>"

with gr.Blocks(css=open("style.css").read()) as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
        <header class="app-header">
            <h1>Waste Segregation Identifier</h1>
            <p class="subtitle">Upload an image to identify if it's Recyclable, Organic, Hazardous, or General Waste.</p>
            <div class="bins-legend">
                <span class="legend-item recyclable">Recyclable</span>
                <span class="legend-item organic">Organic</span>
                <span class="legend-item hazardous">Hazardous</span>
                <span class="legend-item general">General</span>
            </div>
        </header>
        """)

        with gr.Row(elem_classes="file-input-wrapper"):
            image_input = gr.Image(type="numpy", label=None, elem_id="file-input")
            clear_btn = gr.ClearButton([image_input])

        output_html = gr.HTML(elem_id="result-text")

        gr.HTML("""
        <section class="tips">
            <h2 class="section-title">Segregation tips</h2>
            <ul class="tips-list">
                <li><span class="dot organic"></span>Food scraps and yard waste go to Organic.</li>
                <li><span class="dot recyclable"></span>Clean paper, glass, and plastics go to Recyclable.</li>
                <li><span class="dot hazardous"></span>Batteries, chemicals, e-waste are Hazardous.</li>
                <li><span class="dot general"></span>Soiled or mixed items go to General.</li>
            </ul>
        </section>

        <section class="categories-grid">
            <article class="category-card recyclable">
                <div class="card-icon">♻️</div>
                <h3>Recyclable</h3>
                <p>Paper, cardboard, metals, clear plastics, and clean glass.</p>
            </article>
            <article class="category-card organic">
                <div class="card-icon">🌿</div>
                <h3>Organic</h3>
                <p>Fruit peels, leftovers, yard trimmings, coffee grounds.</p>
            </article>
            <article class="category-card hazardous">
                <div class="card-icon">⚠️</div>
                <h3>Hazardous</h3>
                <p>Batteries, paint, chemicals, bulbs. Handle with care.</p>
            </article>
            <article class="category-card general">
                <div class="card-icon">🗑️</div>
                <h3>General</h3>
                <p>Non-recyclable, non-organic items and mixed materials.</p>
            </article>
        </section>
        """)

    image_input.upload(
        fn=predict_image,
        inputs=image_input,
        outputs=output_html
    )

demo.launch()
