import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader
import io

summarizer = pipeline("text2text-generation", model="iamseyhmus7/Turkish-Summarization")

MAX_CHARS = 1500  # Maksimum özetlenecek karakter

def pdf_to_text(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return ""

def summarize_input(text, pdf_file):
    if pdf_file is not None:
        pdf_bytes = pdf_file.read()
        text = pdf_to_text(pdf_bytes)
    if not text or len(text.strip()) < 10:
        return "Özetlenecek yeterli metin bulunamadı."
    # Çok uzun metinleri kırp
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    result = summarizer(
        text,
        max_length=1000,
        min_length=100,
        do_sample=False,
        num_beams=3,
        length_penalty=1.0,
        no_repeat_ngram_size=2
    )
    summary = result[0].get('summary_text') or result[0].get('generated_text') or str(result[0])
    return summary

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
