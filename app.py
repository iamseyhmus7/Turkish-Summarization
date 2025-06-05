import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader
import io

# Hugging Face pipeline ile Türkçe özetleme modeli yükle
summarizer = pipeline("text2text-generation", model="iamseyhmus7/Turkish-Summarization")

def pdf_to_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def summarize_input(text, pdf_file):
    # Öncelik PDF: Eğer PDF varsa onu kullan, yoksa metni kullan
    if pdf_file is not None:
        pdf_bytes = pdf_file.read()
        text = pdf_to_text(pdf_bytes)
    if not text or len(text.strip()) < 10:
        return "Özetlenecek yeterli metin bulunamadı."
    # Çok uzun metinleri kırpabilirsin (isteğe bağlı)
    result = summarizer(text, max_length=80, min_length=20, do_sample=False)
    return result[0]['summary_text']

demo = gr.Interface(
    fn=summarize_input,
    inputs=[
        gr.Textbox(label="Türkçe Metin (isteğe bağlı, PDF yüklemezsen kullanılır)", lines=8, placeholder="Metni buraya yapıştırın..."),
        gr.File(label="PDF Dosyası (isteğe bağlı)")
    ],
    outputs=gr.Textbox(label="Özet"),
    title="Türkçe Metin ve PDF Özetleme",
    description="PDF veya metin yükleyerek otomatik Türkçe özet oluşturun.",
    allow_flagging='never'
)

if __name__ == "__main__":
    demo.launch()
