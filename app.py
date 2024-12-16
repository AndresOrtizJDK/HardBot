#app.py
from flask import Flask, request
import services
from database import inicializar_db
from ml_service import entrenar_modelo_ml

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def recibir_mensajes():
    try:
        body = request.get_json()
        entry = body['entry'][0]
        changes = entry['changes'][0]
        value = changes['value']
        message = value['messages'][0]
        number = services.replace_start(message['from'])
        messageId = message['id']
        contacts = value['contacts'][0]
        name = contacts['profile']['name']
        text = services.obtener_Mensaje_whatsapp(message)

        services.administrar_chatbot(text, number, messageId, name)
        return 'enviado'
    
    except Exception as e:
        return 'no enviado ' + str(e)

if __name__ == '__main__':
    inicializar_db()
    entrenar_modelo_ml()
    app.run()
