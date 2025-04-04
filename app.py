from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

app = Flask(__name__)

# Configuración de modelos (usando versiones más ligeras y disponibles)
try:
    # Modelo para generación de preguntas
    qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
    qa_pipeline = pipeline(
        "text2text-generation",
        model=qg_model,
        tokenizer=qg_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Modelo para otras tareas (usamos t5-small que es más ligero y disponible)
    t5_model_name = "t5-small"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
    
    print("Modelos cargados exitosamente")
except Exception as e:
    print(f"Error al cargar modelos: {str(e)}")
    qa_pipeline = None
    t5_model = None

# Función mejorada para generación de preguntas
def generar_preguntas_calidad(texto, num_preguntas=3):
    if not qa_pipeline:
        return ["Error: Modelo de preguntas no disponible"]
    
    oraciones = [s.strip() for s in re.split(r'[.!?]', texto) if len(s.split()) > 5]
    preguntas_unicas = []
    
    for oracion in oraciones[:10]:  # Limitar a 10 oraciones para eficiencia
        try:
            inputs = f"generate Spanish question: {oracion}"
            outputs = qa_pipeline(
                inputs,
                max_length=100,
                num_beams=4,
                num_return_sequences=2,
                temperature=0.8,
                top_p=0.9
            )
            
            for output in outputs:
                pregunta = output['generated_text'].strip()
                pregunta = re.sub(r'^\s*[¿]?\s*', '¿', pregunta)
                if not pregunta.endswith('?'):
                    pregunta += '?'
                
                if (len(pregunta) > 10 and 
                    pregunta.lower() not in [p.lower() for p in preguntas_unicas]):
                    preguntas_unicas.append(pregunta.capitalize())
                    
        except Exception as e:
            print(f"Error generando preguntas: {str(e)}")
            continue
    
    return preguntas_unicas[:num_preguntas]

# Endpoints con mejor manejo de errores
@app.route('/resumen', methods=['POST'])
def resumen():
    if not t5_model:
        return jsonify({"error": "Modelo no disponible. Intente más tarde."}), 503
        
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "Datos JSON no proporcionados"}), 400
        
    texto = datos.get("texto", "")
    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400

    try:
        entrada = f"summarize in Spanish: {texto}"
        entrada_tokenizada = t5_tokenizer(entrada, return_tensors="pt", truncation=True).input_ids
        salida_tokenizada = t5_model.generate(entrada_tokenizada, max_length=150)
        resumen = t5_tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)
        return jsonify({"resultado": resumen})
    except Exception as e:
        return jsonify({"error": f"Error al generar resumen: {str(e)}"}), 500

@app.route('/traducir', methods=['POST'])
def traducir():
    if not t5_model:
        return jsonify({"error": "Modelo no disponible. Intente más tarde."}), 503
        
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "Datos JSON no proporcionados"}), 400
        
    texto = datos.get("texto", "")
    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400

    try:
        entrada = f"translate English to French: {texto}"  # Eliminé 'informal' para mayor compatibilidad
        entrada_tokenizada = t5_tokenizer(entrada, return_tensors="pt", truncation=True).input_ids
        salida_tokenizada = t5_model.generate(entrada_tokenizada, max_length=150)
        traduccion = t5_tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)
        return jsonify({"resultado": traduccion})
    except Exception as e:
        return jsonify({"error": f"Error al traducir: {str(e)}"}), 500

@app.route('/pregunta', methods=['POST'])
def pregunta():
    if not t5_model:
        return jsonify({"error": "Modelo no disponible. Intente más tarde."}), 503
        
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "Datos JSON no proporcionados"}), 400
        
    pregunta_text = datos.get("pregunta", "")
    contexto = datos.get("contexto", "")
    if not pregunta_text or not contexto:
        return jsonify({"error": "Pregunta y contexto requeridos"}), 400

    try:
        entrada = f"question: {pregunta_text} context: {contexto}"
        entrada_tokenizada = t5_tokenizer(entrada, return_tensors="pt", truncation=True).input_ids
        salida_tokenizada = t5_model.generate(entrada_tokenizada, max_length=150)
        respuesta = t5_tokenizer.decode(salida_tokenizada[0], skip_special_tokens=True)
        return jsonify({"resultado": respuesta})
    except Exception as e:
        return jsonify({"error": f"Error al responder pregunta: {str(e)}"}), 500

@app.route('/generar_preguntas', methods=['POST'])
def generar_preguntas_endpoint():
    if not qa_pipeline:
        return jsonify({"error": "Modelo de preguntas no disponible. Intente más tarde."}), 503
        
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "Datos JSON no proporcionados"}), 400
        
    texto = datos.get("texto", "")
    if not texto or len(texto.split()) < 20:
        return jsonify({"error": "Texto debe tener al menos 20 palabras"}), 400

    try:
        preguntas = generar_preguntas_calidad(texto)
        return jsonify({"preguntas_generadas": preguntas})
    except Exception as e:
        return jsonify({"error": f"Error al generar preguntas: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)