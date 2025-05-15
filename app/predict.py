from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load trained model
model = load_model("model/trash_classifier.h5")

# 🔒 FIXED: This label order matches `train_generator.class_indices`
class_index_map = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

async def predict_trash(file):
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)

        # Predict
        prediction = model.predict(processed_img)[0]
        top_index = int(np.argmax(prediction))
        top_label = class_index_map[top_index]
        top_conf = float(prediction[top_index])

        # Format all probabilities
        class_probs = [
            {
                "class": class_index_map[i],
                "confidence": float(f"{prob:.4f}")
            }
            for i, prob in sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
        ]

        return {
            "predicted_class": top_label,
            "confidence": float(f"{top_conf:.4f}"),
            "all_probabilities": class_probs
        }

    except Exception as e:
        return {"error": str(e)}
