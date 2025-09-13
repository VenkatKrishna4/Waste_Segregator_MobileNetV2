# evaluate.py
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model("best_model.h5")
with open("labels.json") as f:
    labels = json.load(f)
label_list = [labels[str(i)] for i in range(len(labels))]

IMG_SIZE = 224
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory("waste_raw/waste_dataset/val", target_size=(IMG_SIZE,IMG_SIZE), batch_size=32, class_mode='categorical', shuffle=False)

preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print(classification_report(y_true, y_pred, target_names=label_list,labels=np.arange(len(label_list))))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_list, yticklabels=label_list)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()
