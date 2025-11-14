import os
import torch
from PIL import Image, ImageEnhance
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer as AutoTokenizer2
)
from gtts import gTTS
import pygame
import cv2
import uuid
import io
import base64
import warnings

warnings.filterwarnings('ignore')

class MultilingualImageToSpeech:
    def __init__(self, audio_output_dir="outputs/audio_clips"):
        print("Initializing Enhanced Multilingual Image-to-Speech Model...")
        os.makedirs(audio_output_dir, exist_ok=True)
        self.audio_output_dir = audio_output_dir


        pygame.mixer.init()

        # Load BLIP model
        print("Loading BLIP model...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        # Load ViT-GPT2 model
        try:
            print("Loading ViT-GPT2 model...")
            self.vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vit_tokenizer = AutoTokenizer2.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.use_vit = True
        except Exception as e:
            print(f"ViT-GPT2 model not available: {e}")
            self.use_vit = False

        # Language codes
        self.lang_codes = {
            'english': 'en',
            'hindi': 'hi',
            'bengali': 'bn',
            'telugu': 'te',
            'tamil': 'ta',
            'malayalam': 'ml'
        }

        self.tts_supported = {'en', 'hi', 'bn', 'ta', 'te', 'ml'}
        self.supported_languages = list(self.lang_codes.keys())

        # Translation models
        self.translation_models = {}
        self.translation_tokenizers = {}
        self.init_translation_models()
        self.load_nllb_models()

        print("Model initialization complete!")

    def init_translation_models(self):
        model_mappings = {
            'hindi': "Helsinki-NLP/opus-mt-en-hi",
            'malayalam': "Helsinki-NLP/opus-mt-en-ml"
        }
        for lang, model_name in model_mappings.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.translation_tokenizers[lang] = tokenizer
                self.translation_models[lang] = model
                print(f"✅ Loaded {lang} translation model")
            except Exception as e:
                print(f"⚠ Could not load {lang} model: {e}")
                self.translation_tokenizers[lang] = None
                self.translation_models[lang] = None

    def load_nllb_models(self):
        try:
            self.nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.nllb_lang_codes = {
                'bengali': 'ben_Beng',
                'telugu': 'tel_Telu',
                'tamil': 'tam_Taml'
            }
            print("✅ Loaded NLLB translation model")
        except Exception as e:
            print(f"⚠ Could not load NLLB model: {e}")

    def enhance_image_quality(self, image):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        return image

    def generate_caption_blip(self, image):
        try:
            inputs = self.caption_processor(image, return_tensors="pt")
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except:
            return None

    def generate_caption_vit(self, image):
        if not self.use_vit:
            return None
        try:
            pixel_values = self.vit_processor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                output_ids = self.vit_model.generate(pixel_values, max_length=50, num_beams=4)
            caption = self.vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption
        except:
            return None

    def generate_caption(self, image):
        image = self.enhance_image_quality(image)
        captions = [self.generate_caption_blip(image), self.generate_caption_vit(image)]
        captions = [c for c in captions if c]
        if captions:
            return max(captions, key=lambda x: len(x.split()))
        return "Unable to describe the image."

    def translate_text(self, text, target_language):
        lang_key = target_language.lower()
        if lang_key in self.translation_models and self.translation_models[lang_key]:
            tokenizer = self.translation_tokenizers[lang_key]
            model = self.translation_models[lang_key]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=4)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif lang_key in getattr(self, 'nllb_lang_codes', {}):
            tgt_lang = self.nllb_lang_codes[lang_key]
            tokenizer = self.nllb_tokenizer
            model = self.nllb_model
            tokenizer.src_lang = "eng_Latn"
            encoded = tokenizer(text, return_tensors="pt")
            generated = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        else:
            return text

    def generate_tts_audio(self, text, language_code, filename):
        if language_code not in self.tts_supported:
            return None
        tts = gTTS(text=text, lang=language_code, slow=False)
        audio_path = os.path.join(self.audio_output_dir, filename)
        tts.save(audio_path)
        return audio_path

    def play_audio(self, audio_path):
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except:
            pass

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            max_size = 768
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return image
        except:
            return None

    def process_image_to_speech(self, image_path, languages=None):
        results = {}
        if languages is None:
            languages = self.supported_languages
        image = self.preprocess_image(image_path)
        if image is None:
            print("❌ Image not found or invalid.")
            return results

        print(f"\n🖼️ Processing image: {image_path}")
        english_caption = self.generate_caption(image)
        print(f"\n📝 Final English Caption: {english_caption}")

        for lang in languages:
            print(f"\n🔄 Processing {lang.title()}...")
            translation = self.translate_text(english_caption, lang)
            print(f"🌍 {lang.title()} Translation: {translation}")

            lang_code = self.lang_codes.get(lang.lower(), 'en')
            filename = f"{lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
            audio_path = self.generate_tts_audio(translation, lang_code, filename)
            print(f"🔊 {lang.title()} Audio Path: {audio_path}")

            self.play_audio(audio_path)

            results[lang] = {
                'text': translation,
                'audio_path': audio_path
            }

        return results

    def capture_image_from_camera(self):
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture, ESC to exit")
        while True:
            ret, frame = cap.read()
            cv2.imshow("Camera - Press SPACE to capture", frame)
            key = cv2.waitKey(1)
            if key % 256 == 27:  # ESC
                print("Escape pressed, closing camera.")
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key % 256 == 32:  # SPACE
                filename = f"camera_{uuid.uuid4().hex[:8]}.jpg"
                path = os.path.join("outputs", filename)
                cv2.imwrite(path, frame)
                print(f"Captured image saved at {path}")
                cap.release()
                cv2.destroyAllWindows()
                return path

if __name__ == "__main__":
    model = MultilingualImageToSpeech()

    print("\nChoose Input Option:")
    print("1. Capture image from camera")
    print("2. Use existing image file path")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        image_path = model.capture_image_from_camera()
        if image_path:
            model.process_image_to_speech(image_path)
    elif choice == '2':
        image_path = input("Enter full image path: ").strip()
        if os.path.exists(image_path):
            model.process_image_to_speech(image_path)
        else:
            print("❌ Image file not found!")
    else:
        print("❌ Invalid choice!")
