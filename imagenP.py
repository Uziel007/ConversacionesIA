from diffusers import StableDiffusionPipeline
import torch

# Verificar si CUDA está disponible
print("¿CUDA disponible?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de GPU:", torch.cuda.get_device_name(0))
else:
    print("Estás usando CPU 😢")

# Cargar el modelo preentrenado desde un directorio de caché local
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="./modelo_diffusion"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Definir el prompt para la generación de la imagen
prompt = "Un perro y un gato"

# Generar la imagen con menos pasos para mayor velocidad
image = pipeline(
    prompt,
    height=384,  # tamaño más pequeño que 512x512
    width=384,
    num_inference_steps=20  # menos pasos = más rápido
).images[0]

# Guardar la imagen
image.save("imagen_generada.png")

# Mostrar la imagen
image.show()
