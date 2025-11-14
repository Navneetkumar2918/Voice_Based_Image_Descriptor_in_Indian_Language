# 🔍 Hazardous Object Detection with Multilingual Voice Output

This model detects hazardous objects in an image (e.g., knife, scissors, screwdriver) using a trained YOLOv8 model and generates **voice-based warnings** in **multiple Indian languages** including:

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Tamil
- 🇮🇳 Assamese (text only)
- 🇮🇳 Bengali
- 🇮🇳 Telugu
- 🇮🇳 Malayalam

> ✨ Built for visually impaired users to hear real-time feedback about dangerous objects in their environment.
>

## 🚀 Features

- 🔍 **YOLOv8-based object detection**
- 🗣️ **Google Translate + gTTS speech output**
- 🛡️ Hazard level estimation via object **distance + position**
- ✅ Modular and runs offline (except translation/TTS)
- 👩‍🦯 Focused on accessibility for the visually impaired

  ## 📁 Folder Structure

```bash
hazardous-object-detection/
│
├── best.pt                      # YOLOv8 trained weights
├── detect_and_caption.py       # Main script
├── test_images/                # Input images
│   └── img2.jpg
├── outputs/
│   └── audio_clips/            # Translated MP3 clips
├── requirements.txt
└── README.md
```
##🛠️ Setup & Usage

1. Install dependencies
pip install -r requirements.txt

2. Add model weights
Place your trained YOLOv8 model file as best.pt in the root folder.

3. Run detection and TTS
python detect_and_caption.py

