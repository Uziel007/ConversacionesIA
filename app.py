from flask import Flask, request, jsonify, send_file
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
import re
import os
from collections import OrderedDict
from io import BytesIO
from PIL import Image
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preguntas')
def preguntas_vista():
    return render_template('preguntas.html')

@app.route('/resumen')
def resumen_vista():
    return render_template('resumen.html')

@app.route('/traducir')
def traducir_vista():
    return render_template('traducir.html')

@app.route('/responder')
def responder_vista():
    return render_template('responder.html')

@app.route('/imagen')
def imagen_vista():
    return render_template('imagen.html')

# Cargar modelo de generación de preguntas
try:
    qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
    qa_pipeline = pipeline("text2text-generation", model=qg_model, tokenizer=qg_tokenizer,
                           device=0 if torch.cuda.is_available() else -1)
    
    t5_model_name = "t5-small"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    # Modelo de traducción EN -> ES
    translator_pipeline = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    
    print("✅ Modelos de texto cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar modelos de texto: {str(e)}")
    qa_pipeline = None
    t5_model = None
    translator_pipeline = None

# Cargar modelo de generación de imágenes
try:
    cache_dir = "./modelo_diffusion"
    use_local = os.path.exists(cache_dir)

    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        cache_dir=cache_dir if use_local else None,
        local_files_only=use_local
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Modelo de imágenes cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar modelo de imágenes: {str(e)}")
    sd_pipeline = None

# Traducción usando Helsinki-NLP
def traducir_ingles_a_espanol(texto):
    try:
        if not translator_pipeline:
            return texto
        traduccion = translator_pipeline(texto, max_length=150)[0]['translation_text']
        return traduccion
    except Exception as e:
        print(f"Error al traducir: {str(e)}")
        return texto

def son_preguntas_similares(p1, p2):
    p1 = p1.lower().replace('¿', '').replace('?', '')
    p2 = p2.lower().replace('¿', '').replace('?', '')
    set1 = set(p1.split())
    set2 = set(p2.split())
    interseccion = len(set1 & set2)
    union = len(set1 | set2)
    return (interseccion / union) > 0.6 if union else False

def filtrar_preguntas_similares(lista):
    unicas = []
    for p in lista:
        if not any(son_preguntas_similares(p, u) for u in unicas):
            unicas.append(p)
    return unicas

def generar_preguntas_calidad(texto, num_preguntas=3):
    if not qa_pipeline:
        return ["Error: Modelo no cargado correctamente"]

    oraciones = [s.strip() for s in re.split(r'[.!?]', texto) if len(s.split()) > 5]
    todas = []

    for o in oraciones[:15]:
        try:
            input_text = f"generate question: {o}"
            outputs = qa_pipeline(input_text, max_length=100, num_beams=4, num_return_sequences=2,
                                  temperature=0.9, top_p=0.95, do_sample=True)
            for out in outputs:
                pregunta_en = out['generated_text'].strip()
                pregunta_es = traducir_ingles_a_espanol(pregunta_en).strip()
                if not pregunta_es.endswith('?'):
                    pregunta_es += '?'
                pregunta_es = re.sub(r'^\s*[¿]?\s*', '¿', pregunta_es).capitalize()
                todas.append(pregunta_es)
        except Exception as e:
            print(f"Error generando preguntas: {str(e)}")

    unicas = list(OrderedDict.fromkeys(todas))
    filtradas = filtrar_preguntas_similares(unicas)

    if len(filtradas) < num_preguntas:
        try:
            input_text = f"generate {num_preguntas} diverse questions about: {texto[:500]}"
            outputs = qa_pipeline(input_text, max_length=150, num_beams=4, num_return_sequences=1, temperature=0.8)
            extra = outputs[0]['generated_text'].split('\n')
            extra_es = [traducir_ingles_a_espanol(p.strip()) for p in extra if len(p.strip().split()) > 3]
            filtradas.extend(extra_es)
            filtradas = filtrar_preguntas_similares(list(OrderedDict.fromkeys(filtradas)))
        except Exception as e:
            print(f"Error generando preguntas extra: {str(e)}")

    return filtradas[:num_preguntas]

@app.route('/generar_preguntas', methods=['POST'])
def generar_preguntas_endpoint():
    try:
        datos = request.json
        texto = datos.get("texto", "")
        if not texto or len(texto.split()) < 20:
            return jsonify({"error": "El texto debe tener al menos 20 palabras"}), 400

        preguntas = generar_preguntas_calidad(texto)
        while len(preguntas) < 3 and len(preguntas) > 0:
            preguntas.append(preguntas[-1])
        return jsonify({"preguntas_generadas": preguntas[:3]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resumen', methods=['POST'])
def resumen():
    if not t5_model:
        return jsonify({"error": "Modelo no disponible"}), 503
    datos = request.get_json()
    texto = datos.get("texto", "")
    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400
    try:
        entrada = f"summarize in Spanish: {texto}"
        tokens = t5_tokenizer(entrada, return_tensors="pt", truncation=True).input_ids
        salida = t5_model.generate(tokens, max_length=150)
        resumen = t5_tokenizer.decode(salida[0], skip_special_tokens=True)
        return jsonify({"resultado": resumen})
    except Exception as e:
        return jsonify({"error": f"Error al generar resumen: {str(e)}"}), 500

@app.route('/traducir', methods=['POST'])
def traducir():
    datos = request.get_json()
    texto = datos.get("texto", "")
    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400
    try:
        resultado = traducir_ingles_a_espanol(texto)
        return jsonify({"resultado": resultado})
    except Exception as e:
        return jsonify({"error": f"Error al traducir: {str(e)}"}), 500

@app.route('/pregunta', methods=['POST'])
def pregunta():
    if not t5_model:
        return jsonify({"error": "Modelo no disponible"}), 503
    datos = request.get_json()
    pregunta_text = datos.get("pregunta", "")
    contexto = datos.get("contexto", "")
    if not pregunta_text or not contexto:
        return jsonify({"error": "Pregunta y contexto requeridos"}), 400
    try:
        entrada = f"question: {pregunta_text} context: {contexto}"
        tokens = t5_tokenizer(entrada, return_tensors="pt", truncation=True).input_ids
        salida = t5_model.generate(tokens, max_length=150)
        respuesta = t5_tokenizer.decode(salida[0], skip_special_tokens=True)
        return jsonify({"resultado": respuesta})
    except Exception as e:
        return jsonify({"error": f"Error al responder pregunta: {str(e)}"}), 500

@app.route('/generar_imagen', methods=['POST'])
def generar_imagen():
    if not sd_pipeline:
        return jsonify({"error": "Modelo de generación de imágenes no disponible"}), 503
    datos = request.get_json()
    prompt = datos.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt no proporcionado"}), 400
    try:
        height = min(int(datos.get("height", 384)), 768)
        width = min(int(datos.get("width", 384)), 768)
        steps = min(int(datos.get("steps", 20)), 50)

        image = sd_pipeline(prompt, height=height, width=width, num_inference_steps=steps).images[0]
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Error al generar imagen: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
