# 🇹🇷 Turkish Summarization with mT5

Bu proje, **Türkçe metin özetleme** görevini gerçekleştirmek üzere **mT5-small** modelinin **tam (full) fine-tuning** yöntemiyle eğitilmesiyle geliştirilmiştir.  
Eğitimde, Hugging Face üzerinde yer alan [`gullnihal/mlsum_tr`](https://huggingface.co/datasets/gullnihal/mlsum_tr) veri seti kullanılmıştır.

## 🚀 Özellikler
- **mT5-small** modeli ile Türkçe metin özetleme
- Hem **metin girdisi** hem de **PDF belgelerinin** özetlenmesi
- Hugging Face Spaces üzerinde canlı demo
- Basit ve hızlı kullanım

## 📂 Veri Seti
- Kaynak: [`gullnihal/mlsum_tr`](https://huggingface.co/datasets/gullnihal/mlsum_tr)
- Türkçe haber metinlerinden oluşur
- Eğitim/Doğrulama/Test ayrımı hazır

## 🛠️ Eğitim Süreci
- **Model:** `google/mt5-small`
- **Fine-tuning:** Tam model ağırlıkları güncellendi (full fine-tuning)
- **Amaç:** Uzun Türkçe metinlerin anlam bütünlüğünü koruyarak kısa özetler üretmek
- **Donanım:** Google Colab T4 / A100

## 📦 Kullanım
### Hugging Face ile
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "iamseyhmus7/turkish-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Türkiye'nin ilk astronotu Alper Gezeravcı, ISS'teki görevini tamamladı."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(**inputs, max_length=64, min_length=10)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
