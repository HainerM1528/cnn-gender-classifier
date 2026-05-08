import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Clasificador de Género CNN", layout="centered")
st.title("Clasificador de Género — CNN + XAI")
st.markdown("Sube una imagen de un rostro y el modelo predecirá el género con mapas de interpretabilidad.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/Users/hainermejia/models/model.keras")

model = load_model()

def get_saliency(model, img_array):
    img_tensor = tf.Variable(img_array[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        pred = model(img_tensor, training=False)
    grads = tape.gradient(pred, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    p99 = np.percentile(saliency, 99)
    saliency = np.clip(saliency, 0, p99)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def get_gradcam(model, img_array):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', name='last_conv')(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model_func = tf.keras.Model(inputs, outputs)
    model_func.set_weights(model.get_weights())

    feature_extractor = tf.keras.Model(
        inputs=model_func.input,
        outputs=[model_func.get_layer('last_conv').output, model_func.output]
    )
    img_const = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_const)
        conv_outputs, preds = feature_extractor(img_const, training=False)
        loss = tf.reduce_sum(preds)
    grads_cam = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads_cam, axis=(0, 1))
    heatmap = (conv_outputs[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return cv2.resize(heatmap, (224, 224))

uploaded_file = st.file_uploader("Sube una imagen de un rostro", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0

    st.image(image, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando..."):
        pred = model.predict(img_resized[np.newaxis, ...])[0][0]
        label = "Female" if pred > 0.5 else "Male"
        confianza = pred if pred > 0.5 else 1 - pred

        st.markdown("---")
        st.subheader("Predicción")
        col1, col2 = st.columns(2)
        col1.metric("Género", label)
        col2.metric("Confianza", f"{confianza * 100:.1f}%")

        st.markdown("---")
        st.subheader("Mapas de Interpretabilidad")

        saliency = get_saliency(model, img_resized)
        heatmap = get_gradcam(model, img_resized)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title("Saliency Map")
        axes[1].axis("off")
        axes[2].imshow(img_resized)
        axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[2].set_title("Grad-CAM")
        axes[2].axis("off")
        plt.tight_layout()
        st.pyplot(fig)