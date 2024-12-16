#ml_service.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def entrenar_modelo_ml():
    intentos = [
        "quiero registrar una cita", "deseo ver mis citas",
        "muéstrame el catálogo de precios", "agenda una cita",
        "envíame el catálogo", "quiero saber más"
    ]
    etiquetas = ["registrar_cita", "ver_citas", "catalogo", "registrar_cita", "catalogo", "ayuda"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(intentos)

    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    with open("modelo_intenciones.pkl", "wb") as f:
        pickle.dump((vectorizer, modelo), f)

def cargar_modelo_ml():
    with open("modelo_intenciones.pkl", "rb") as f:
        vectorizer, modelo_ml = pickle.load(f)
    return vectorizer, modelo_ml

# Carga explícita del modelo en las funciones que lo necesitan
def predecir_intencion(mensaje):
    vectorizer, modelo_ml = cargar_modelo_ml()
    X = vectorizer.transform([mensaje])
    return modelo_ml.predict(X)[0]
