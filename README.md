# 🇹🇷 Turkish Summarization with mT5

Bu proje, **Türkçe metin özetleme** görevini gerçekleştirmek üzere **mT5-small** modelinin **tam (full) fine-tuning** yöntemiyle eğitilmesiyle geliştirilmiştir.  
Eğitimde, Hugging Face üzerinde yer alan [`gullnihal/mlsum_tr`](https://huggingface.co/datasets/gullnihal/mlsum_tr) veri seti kullanılmıştır.  
Canlı olarak denemek için [🌐 Hugging Face Space - Turkish Summarization](https://huggingface.co/spaces/iamseyhmus7/turkish-summarization) sayfasını ziyaret edebilirsiniz.

## 🚀 Özellikler
- **mT5-small** modeli ile yüksek kaliteli Türkçe metin özetleme
- Hem **metin girdisi** hem de **PDF belgelerinin** özetlenmesi
- Hugging Face Spaces üzerinde web arayüzlü canlı demo
- Basit, hızlı ve Docker ile taşınabilir kullanım

## 📂 Veri Seti
- Kaynak: [`gullnihal/mlsum_tr`](https://huggingface.co/datasets/gullnihal/mlsum_tr)
- Türkçe haber metinlerinden oluşmaktadır
- Eğitim/Doğrulama/Test ayrımı hazır

## 🛠️ Eğitim Süreci
- **Model:** `google/mt5-small`
- **Fine-tuning:** Tam model ağırlıkları güncellendi (full fine-tuning)
- **Amaç:** Uzun Türkçe metinlerin anlam bütünlüğünü koruyarak kısa özetler üretmek
- **Donanım:** Google Colab T4 / A100

## 🐳 Docker Kullanımı
Proje, **Docker** kullanılarak kolayca herhangi bir ortamda çalıştırılabilir hale getirilmiştir.  
Tüm bağımlılıklar ve çalışma ortamı `Dockerfile` içinde tanımlanmıştır. Bu sayede:
- Ortam uyumsuzluğu sorunları ortadan kalkar
- Tek komutla kurulum ve çalıştırma yapılır
- Proje bulut sunuculara kolayca taşınabilir

## 📦 Kullanım
### Hugging Face ile
```python

from transformers import pipeline
# pipe özellikleri değiştirilebilir.
pipe = pipeline("text2text-generation", model="iamseyhmus7/Turkish-Summarization")
gen_kwargs = {
    "length_penalty": 1.0,
    "num_beams": 4,
    "max_length": 1000,
    "min_length": 50,
    "no_repeat_ngram_size": 2
}

metin = """
Sonbahar mevsimi geldiğinde doğa büyüleyici bir değişim yaşar.
Ağaçların yaprakları sarı, turuncu ve kırmızı tonlarına bürünerek adeta bir renk cümbüşü oluşturur.
Hafif esen rüzgar, yerdeki yaprakları savururken temiz ve serin bir hava hissedilir.
İnsanlar kalın kıyafetlerini giymeye başlar, sıcak içecekler eşliğinde keyifli sohbetler eder.
Doğa, kışa hazırlanırken dingin ve huzur verici bir atmosfer sunar. Bu mevsimin getirdiği sakinlik,
hem ruhu dinlendirir hem de yeni başlangıçlar için ilham kaynağı olur.
"""

output = pipe(metin, **gen_kwargs)
print(output[0]['generated_text'])
revini tamamladı."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(**inputs, max_length=64, min_length=10)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
