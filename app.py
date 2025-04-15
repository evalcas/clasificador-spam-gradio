import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# Dataset de ejemplo
correos = [
    "¡Gana dinero fácil ahora!",
    "Reunión importante mañana",
    "Haz clic para reclamar tu premio",
    "Te envío el informe solicitado",
    "Compra ahora con 70% de descuento",
    "Hola, ¿puedes revisar esto por favor?",
    "Actualización importante de tu cuenta",
    "Oferta limitada solo hoy",
    "Gracias por tu ayuda ayer",
    "Obtén un préstamo sin requisitos"
]
etiquetas = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]

# Preprocesamiento
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(correos)
y = etiquetas

# Modelo
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Función con probabilidad
def clasificar_correo(correo):
    vector = vectorizador.transform([correo])
    prediccion = modelo.predict(vector)[0]
    probabilidad = modelo.predict_proba(vector)[0][1]  # Probabilidad de clase "1" (spam)
    resultado = "🚫 Spam" if prediccion == 1 else "✅ No es Spam"
    return f"{resultado}\n\n📊 Probabilidad de spam: {probabilidad:.2%}"

# Interfaz
iface = gr.Interface(
    fn=clasificar_correo,
    inputs=gr.Textbox(lines=5, placeholder="Escribe el contenido del correo aquí..."),
    outputs="text",
    title="Clasificador de Correos Spam",
    description="Clasifica correos como spam o no spam con probabilidad usando un árbol de decisión.",
    live=False
)

iface.launch()
