from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Cargar el modelo T5
nombre_modelo = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

@app.route('/resumen', methods=['POST'])
def resumen():
    datos = request.json
    texto = datos.get("texto", "")

    if not texto:
        return jsonify({"error": "No se proporcionó texto"}), 400

    entrada = f"summarize: {texto}"
    entrada_tokenizada = tokenizer(entrada, return_tensors="pt").input_ids
    salida_tokenizada = modelo.generate(entrada_tokenizada, max_length=150)
    resumen = tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)

    return jsonify({"resultado": resumen})

@app.route('/traducir', methods=['POST'])
def traducir():
    datos = request.json
    texto = datos.get("texto", "")

    if not texto:
        return jsonify({"error": "No se proporcionó texto"}), 400

    entrada = f"translate English to French: {texto}"
    entrada_tokenizada = tokenizer(entrada, return_tensors="pt").input_ids
    salida_tokenizada = modelo.generate(entrada_tokenizada, max_length=150)
    traduccion = tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)

    return jsonify({"resultado": traduccion})

@app.route('/pregunta', methods=['POST'])
def pregunta():
    datos = request.json
    pregunta = datos.get("pregunta", "")
    contexto = datos.get("contexto", "")

    if not pregunta or not contexto:
        return jsonify({"error": "Se requiere pregunta y contexto"}), 400

    entrada = f"question: {pregunta}. context: {contexto}"
    entrada_tokenizada = tokenizer(entrada, return_tensors="pt").input_ids
    salida_tokenizada = modelo.generate(entrada_tokenizada, max_length=150)
    respuesta = tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)

    return jsonify({"resultado": respuesta})

@app.route('/generar_preguntas', methods=['POST'])
def generar_preguntas():
    datos = request.json
    texto = datos.get("texto", "")

    if not texto:
        return jsonify({"error": "No se proporcionó texto"}), 400

    entrada = f"generate question: {texto}"
    entrada_tokenizada = tokenizer(entrada, return_tensors="pt").input_ids
    salida_tokenizada = modelo.generate(entrada_tokenizada, max_length=150)
    preguntas_generadas = tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)

    return jsonify({"resultado": preguntas_generadas})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
