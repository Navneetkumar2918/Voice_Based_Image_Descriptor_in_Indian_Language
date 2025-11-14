import pyttsx3

try:
    engine = pyttsx3.init(driverName='sapi5')  # Use Windows speech engine
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1.0)  # Volume
    engine.say("Test: Text to speech is working properly.")
    engine.runAndWait()
    engine.stop()
except Exception as e:
    print("[ERROR]:", e)
