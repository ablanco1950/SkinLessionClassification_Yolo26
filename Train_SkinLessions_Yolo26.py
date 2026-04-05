from ultralytics import YOLO

# 1. Cargar un modelo de clasificación preentrenado
# from https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n-cls.pt
model = YOLO('yolo26n-cls.pt') 

# 2. Entrenar con el dataset personalizado
results = model.train(
    data='Dir_SkinCancer_Yolo26', # Apunta a la carpeta raíz del dataset
    epochs=50,                # Número de pasadas completas
    imgsz=224,                # Tamaño de imagen (224 es estándar para clasificación)
    device="cpu"                  # Usar GPU (0) o 'cpu'
)
