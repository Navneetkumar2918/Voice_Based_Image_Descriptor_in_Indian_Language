from ultralytics import YOLO
import cv2
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import time
import pygame
import tempfile
import platform

class MultilingualYOLODetector:
    def __init__(self, model_path, audio_output_dir="outputs/audio_clips"):
        """Initialize the multilingual YOLO detector"""
        self.model_path = model_path
        self.audio_output_dir = audio_output_dir
        os.makedirs(self.audio_output_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # Define supported languages
        self.languages = {
            "Hindi": "hi",
            "Tamil": "ta", 
            "Assamese": "as",
            "Bengali": "bn",
            "Telugu": "te",
            "Malayalam": "ml"
        }
        
        print(f"🔊 Audio output directory: {self.audio_output_dir}")
        print(f"🌍 Supported languages: {list(self.languages.keys())}")
    
    def play_audio_pygame(self, audio_path):
        """Play audio using pygame"""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            return True
        except Exception as e:
            print(f"❌ Pygame audio playback failed: {e}")
            return False
    
    def play_audio_winsound(self, audio_path):
        """Play audio using Windows winsound (Windows only)"""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_FILENAME)
                return True
            return False
        except Exception as e:
            print(f"❌ Winsound audio playback failed: {e}")
            return False
    
    def play_audio_system(self, audio_path):
        """Play audio using system commands"""
        try:
            system = platform.system()
            if system == "Windows":
                os.system(f'start "" "{audio_path}"')
            elif system == "Darwin":  # macOS
                os.system(f'afplay "{audio_path}"')
            elif system == "Linux":
                os.system(f'mpg123 "{audio_path}" > /dev/null 2>&1')
            
            time.sleep(2)  # Give time for audio to play
            return True
        except Exception as e:
            print(f"❌ System audio playback failed: {e}")
            return False
    
    def play_audio(self, audio_path):
        """Play audio using the best available method"""
        methods = [
            self.play_audio_pygame,
            self.play_audio_winsound,
            self.play_audio_system
        ]
        
        for method in methods:
            if method(audio_path):
                return True
        
        print(f"❌ All audio playback methods failed for {audio_path}")
        return False
    
    def detect_objects(self, image_path, confidence_threshold=0.4):
        """Detect objects in image and generate captions"""
        frame = cv2.imread(image_path)
        captions = []
        
        if frame is None:
            print(f"❌ Image not loaded. Check path: {image_path}")
            return captions
        
        print(f"📸 Processing image: {image_path}")
        results = self.model(frame)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls].capitalize()
                
                if conf > confidence_threshold:
                    # Calculate distance based on bounding box area
                    area = (x2 - x1) * (y2 - y1)
                    if area > 100000:
                        distance = "very close"
                    elif area > 50000:
                        distance = "close"
                    elif area > 20000:
                        distance = "medium distance"
                    else:
                        distance = "far"
                    
                    # Calculate position
                    center_x = (x1 + x2) // 2
                    width = frame.shape[1]
                    if center_x < width * 0.33:
                        position = "left"
                    elif center_x < width * 0.66:
                        position = "center"
                    else:
                        position = "right"
                    
                    sentence = f"{class_name} detected on the {position}, and it is {distance}."
                    captions.append({
                        'text': sentence,
                        'confidence': conf,
                        'class': class_name,
                        'position': position,
                        'distance': distance
                    })
        
        return captions
    
    def generate_tts(self, text, language_code, filename):
        """Generate TTS audio file"""
        try:
            tts = gTTS(text=text, lang=language_code, slow=False)
            audio_path = os.path.join(self.audio_output_dir, filename)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return None
    
    def process_image_multilingual(self, image_path, play_audio=True, save_audio=True):
        """Process image with multilingual output"""
        # Detect objects
        captions = self.detect_objects(image_path)
        
        if not captions:
            print("❌ No objects detected in the image.")
            return
        
        print(f"\n🎯 Detected {len(captions)} objects\n")
        
        # Process English captions
        print("🇺🇸 [English Captions]:\n")
        for idx, caption_info in enumerate(captions):
            text = caption_info['text']
            conf = caption_info['confidence']
            print(f"{idx+1}. {text} (Confidence: {conf:.2f})")
            
            if save_audio:
                audio_path = self.generate_tts(text, 'en', f"english_{idx}.mp3")
                if audio_path and play_audio:
                    print(f"🔊 Playing English audio {idx+1}...")
                    self.play_audio(audio_path)
                    time.sleep(0.5)
        
        # Process translated captions
        print("\n🌍 [Translated Captions + TTS]:\n")
        
        for lang_name, lang_code in self.languages.items():
            print(f"\n🗣️ [{lang_name}]")
            
            for idx, caption_info in enumerate(captions):
                english_text = caption_info['text']
                
                try:
                    # Translate text
                    translated = GoogleTranslator(source='auto', target=lang_code).translate(english_text)
                    print(f"{idx+1}. {translated}")
                    
                    if save_audio:
                        # Generate TTS
                        audio_path = self.generate_tts(translated, lang_code, f"{lang_name}_{idx}.mp3")
                        
                        if audio_path and play_audio:
                            print(f"🔊 Playing {lang_name} audio {idx+1}...")
                            self.play_audio(audio_path)
                            time.sleep(0.5)
                
                except Exception as e:
                    print(f"❌ Translation/TTS failed for {lang_name}: {e}")
    
    def batch_process(self, image_folder, play_audio=False, save_audio=True):
        """Process multiple images in a folder"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print(f"❌ No supported image files found in {image_folder}")
            return
        
        print(f"📁 Processing {len(image_files)} images from {image_folder}")
        
        for idx, filename in enumerate(image_files):
            image_path = os.path.join(image_folder, filename)
            print(f"\n{'='*60}")
            print(f"Processing {idx+1}/{len(image_files)}: {filename}")
            print('='*60)
            
            self.process_image_multilingual(image_path, play_audio, save_audio)
    
    def get_detection_summary(self, image_path):
        """Get a summary of detections without audio"""
        captions = self.detect_objects(image_path)
        
        if not captions:
            return "No objects detected."
        
        summary = f"Detected {len(captions)} objects:\n"
        for i, caption in enumerate(captions):
            summary += f"{i+1}. {caption['class']} ({caption['position']}, {caption['distance']}) - {caption['confidence']:.2f}\n"
        
        return summary

def main():
    """Main function for interactive use"""
    print("🚀 Multilingual YOLO Object Detection System")
    print("=" * 50)
    
    # Configuration
    model_path = "best.pt"  # Update this path
    default_image = "test_images/img2.jpg"  # Update this path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure your YOLO model file exists and update the model_path variable.")
        return
    
    # Initialize detector
    try:
        detector = MultilingualYOLODetector(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    while True:
        print("\n📋 Options:")
        print("1. Process single image (with audio)")
        print("2. Process single image (silent)")
        print("3. Batch process folder")
        print("4. Get detection summary")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            img_path = input(f"Enter image path (or press Enter for default: {default_image}): ").strip()
            if not img_path:
                img_path = default_image
            
            if os.path.exists(img_path):
                detector.process_image_multilingual(img_path, play_audio=True, save_audio=True)
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif choice == '2':
            img_path = input(f"Enter image path (or press Enter for default: {default_image}): ").strip()
            if not img_path:
                img_path = default_image
            
            if os.path.exists(img_path):
                detector.process_image_multilingual(img_path, play_audio=False, save_audio=True)
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif choice == '3':
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                play_audio = input("Play audio during processing? (y/n): ").strip().lower() == 'y'
                detector.batch_process(folder_path, play_audio=play_audio)
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        elif choice == '4':
            img_path = input(f"Enter image path (or press Enter for default: {default_image}): ").strip()
            if not img_path:
                img_path = default_image
            
            if os.path.exists(img_path):
                summary = detector.get_detection_summary(img_path)
                print(f"\n📊 Detection Summary:\n{summary}")
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice! Please try again.")

if __name__ == "__main__":
    # Required packages
    print("📦 Required packages:")
    print("pip install ultralytics opencv-python deep-translator gtts pygame")
    print("\nMake sure you have:")
    print("1. A trained YOLO model file (best.pt)")
    print("2. Test images in the specified directory")
    print("3. Internet connection for translation and TTS")
    print("\n" + "="*50)
    
    main()