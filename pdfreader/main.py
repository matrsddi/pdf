import tensorflow as tf
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import cv2
import os
from io import BytesIO

MODEL = tf.keras.models.load_model('model_39.h5')

app = FastAPI()

class UserInput(BaseModel):
    user_input: float

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(UserInput: UserInput):
    prediction = MODEL.predict([UserInput.user_input])
    return {"prediction": float(prediction)}

@app.post('/extract_images/')
async def extract_images(file: UploadFile):
    # كود استخراج الصور من ملف PDF
    import pyttsx3
    import pdfplumber
    from PyPDF4.pdf import PdfFileReader, PdfFileWriter
    from gtts import gTTS
    import fitz
    from PIL import Image

    pdf_bytes = await file.read()  # قراءة الملف من البايتات
    pdf_file = BytesIO(pdf_bytes)

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            filename_txt = f"page_{i+1}.txt"
            with open(filename_txt, 'w', encoding='utf-8') as f:
                f.write(text)
            myobj = gTTS(text=text, lang='en', slow=False)
            filename_mp3 = f"page_{i+1}.mp3"
            myobj.save(filename_mp3)

    def extract_images(pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfFileReader(pdf_file)
            image_files = []
            for i in range(reader.getNumPages()):
                page = reader.getPage(i)
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject']
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            image_file = f'{obj[1:]}-{i+1}.jpg'
                            image_files.append((image_file, i+1))
                            image_dir = f'{pdf_path}_images'
                            if not os.path.exists(image_dir):
                                os.makedirs(image_dir)
                            with open(os.path.join(image_dir, image_file), 'wb') as f:
                                f.write(xObject[obj].getData())
        return image_files

    images = extract_images(pdf_file)
    for image_file, page_num in images:
        print(f'Image {image_file} found on page {page_num}')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
