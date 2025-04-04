from flask import Flask, request, jsonify, send_file
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
import re
import os
from collections import OrderedDict
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuración de modelos de texto
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
    
    # Modelo T5 para otras tareas
    t5_model_name = "t5-small"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    
    print("✅ Modelos de texto cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar modelos de texto: {str(e)}")
    qa_pipeline = None
    t5_model = None

# Configuración del modelo de generación de imágenes
try:
    print("\nConfigurando modelo de generación de imágenes...")
    print(f"¿CUDA disponible? {'✅ Sí' if torch.cuda.is_available() else '❌ No'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Verificar si existe caché local
    cache_dir = "./modelo_diffusion"
    use_local = os.path.exists(cache_dir)
    
    # Cargar el modelo
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        cache_dir=cache_dir if use_local else None,
        local_files_only=use_local
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("✅ Modelo de generación de imágenes cargado correctamente")
    print(f"Usando {'caché local' if use_local else 'descarga remota'}")
except Exception as e:
    print(f"❌ Error al cargar modelo de imágenes: {str(e)}")
    sd_pipeline = None

# Funciones auxiliares para generación de preguntas
def son_preguntas_similares(pregunta1, pregunta2):
    p1 = pregunta1.lower().replace('¿', '').replace('?', '')
    p2 = pregunta2.lower().replace('¿', '').replace('?', '')
    palabras1 = set(p1.split())
    palabras2 = set(p2.split())
    interseccion = len(palabras1 & palabras2)
    union = len(palabras1 | palabras2)
    return (interseccion / union) > 0.6 if union > 0 else False

def filtrar_preguntas_similares(preguntas):
    preguntas_unicas = []
    for pregunta in preguntas:
        es_unica = True
        for pregunta_unica in preguntas_unicas:
            if son_preguntas_similares(pregunta, pregunta_unica):
                es_unica = False
                break
        if es_unica:
            preguntas_unicas.append(pregunta)
    return preguntas_unicas

def generar_preguntas_calidad(texto, num_preguntas=3):
    if not qa_pipeline:
        return ["Error: Modelo no cargado correctamente"]
    
    oraciones = [s.strip() for s in re.split(r'[.!?]', texto) if len(s.split()) > 5]
    todas_preguntas = []
    
    for oracion in oraciones[:15]:
        try:
            inputs = f"generate Spanish question: {oracion}"
            outputs = qa_pipeline(
                inputs,
                max_length=100,
                num_beams=4,
                num_return_sequences=2,
                temperature=0.9,
                top_p=0.95,
                do_sample=True
            )
            
            for output in outputs:
                pregunta = output['generated_text'].strip()
                pregunta = re.sub(r'^\s*[¿]?\s*', '¿', pregunta)
                if not pregunta.endswith('?'):
                    pregunta += '?'
                pregunta = pregunta.capitalize()
                if pregunta.startswith('¿'):
                    pregunta = '¿' + pregunta[1:].capitalize()
                todas_preguntas.append(pregunta)
        except Exception as e:
            print(f"Error generando preguntas: {str(e)}")
            continue
    
    preguntas_unicas = list(OrderedDict.fromkeys(todas_preguntas))
    preguntas_filtradas = filtrar_preguntas_similares(preguntas_unicas)
    
    if len(preguntas_filtradas) < num_preguntas:
        try:
            inputs = f"generate {num_preguntas} diverse Spanish questions about: {texto[:500]}"
            outputs = qa_pipeline(
                inputs,
                max_length=150,
                num_beams=4,
                num_return_sequences=1,
                temperature=0.8
            )
            preguntas_extra = [q.strip() for q in outputs[0]['generated_text'].split('\n') 
                             if q.strip() and len(q.split()) > 3]
            preguntas_filtradas.extend(preguntas_extra)
            preguntas_filtradas = filtrar_preguntas_similares(list(OrderedDict.fromkeys(preguntas_filtradas)))
        except Exception as e:
            print(f"Error generando preguntas extra: {str(e)}")
    
    return preguntas_filtradas[:num_preguntas]

# Endpoints de texto
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
        entrada = f"translate English to French: {texto}"
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

# Nuevo endpoint para generación de imágenes
@app.route('/generar_imagen', methods=['POST'])
def generar_imagen():
    if not sd_pipeline:
        return jsonify({"error": "Modelo de generación de imágenes no disponible"}), 503
        
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "Datos JSON no proporcionados"}), 400
        
    prompt = datos.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt no proporcionado"}), 400

    try:
        # Configuración de la generación
        height = min(int(datos.get("height", 384)), 768)
        width = min(int(datos.get("width", 384)), 768)
        steps = min(int(datos.get("steps", 20)), 50)
        
        # Generar la imagen
        image = sd_pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps
        ).images[0]
        
        # Guardar en buffer de memoria
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        return jsonify({"error": f"Error al generar imagen: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)