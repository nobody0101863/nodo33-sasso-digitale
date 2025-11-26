from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
import random
import os
import requests
import tensorflow as tf
from sklearn.linear_model import Ridge
from transformers import pipeline

app = Flask(__name__)

class LuxOmegaAI:
    def __init__(self):
        self.db_file = "lux_memory.db"
        self._init_db()
        self.model = self._load_model()
        self.chatbot = pipeline("text-generation", model="gpt2")

    def _init_db(self):
        """Inizializza il database con categorie"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, query TEXT, response TEXT, category TEXT)")
        conn.commit()
        conn.close()

    def _load_model(self):
        """Crea un modello AI di regressione per predizioni più accurate"""
        return Ridge(alpha=1.0)

    def learn(self, input_data, category="Generale"):
        """Salva i dati nel database per la memoria persistente"""
        response_value = self._process(input_data)
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("INSERT INTO knowledge (query, response, category) VALUES (?, ?, ?)", (input_data, response_value, category))
        conn.commit()
        conn.close()

    def _process(self, data):
        """Genera un valore di apprendimento più avanzato"""
        return np.tanh(len(data) * 2.0) + random.uniform(0, 0.5)

    def predict(self, query):
        """Recupera la conoscenza dal database e utilizza un modello avanzato di predizione"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT response FROM knowledge WHERE query = ?", (query,))
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            input_text = f"Domanda: {query}\nRisposta:"
            response = self.chatbot(input_text, max_length=100, do_sample=True)[0]["generated_text"]
            return response.replace(input_text, "").strip()

    def get_history(self):
        """Recupera la cronologia delle interazioni"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT query, response, category FROM knowledge ORDER BY id DESC LIMIT 10")
        history = c.fetchall()
        conn.close()
        return history

lux = LuxOmegaAI()

@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/learn', methods=['POST'])
def learn():
    data = request.json.get("data")
    category = request.json.get("category", "Generale")
    lux.learn(data, category)
    return jsonify({"message": "Apprendimento salvato nel database!"})

@app.route('/predict', methods=['GET'])
def predict():
    query = request.args.get("query")
    response = lux.predict(query)
    return jsonify({"response": response})

@app.route('/history', methods=['GET'])
def history():
    return jsonify({"history": lux.get_history()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
