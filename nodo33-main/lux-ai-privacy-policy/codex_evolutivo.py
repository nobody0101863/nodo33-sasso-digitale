#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
import pyttsx3

# Simulazione AI per ottimizzazione del sistema
data = {"CPU_Usage": [20, 35, 50, 65, 80], "RAM_Usage": [2, 4, 6, 8, 10], "Performance_Score": [90, 75, 60, 45, 30]}
df = pd.DataFrame(data)
model = LinearRegression()
model.fit(df[["CPU_Usage", "RAM_Usage"]], df["Performance_Score"])

def analizza_prestazioni():
    status = [[30, 5]]  # Simuliamo 30% CPU e 5GB RAM
    prediction = model.predict(status)
    print(f"🤖 AI: Prestazioni previste: {prediction[0]:.2f} / 100")
    if prediction[0] < 50:
        print("⚠️ Ottimizzazione necessaria.")
        os.system("brew cleanup && killall Dock")
    else:
        print("✅ Sistema stabile!")

def ascolta_comando():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    with sr.Microphone() as source:
        print("🎤 Codex in ascolto...")
        audio = recognizer.listen(source)
        try:
            comando = recognizer.recognize_google(audio)
            print(f"🔊 Hai detto: {comando}")
            if "status" in comando:
                print("📊 Il Codex è attivo e stabile.")
                engine.say("Il Codex è attivo e stabile")
                engine.runAndWait()
        except sr.UnknownValueError:
            print("❌ Non ho capito!")

analizza_prestazioni()
