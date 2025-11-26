from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

def query_database(query, params=()):
    conn = sqlite3.connect("lux_tokens.db")
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    return result

@app.route('/tokens', methods=['GET'])
def get_tokens():
    """📡 Ottieni tutti i token dal database via API"""
    tokens = query_database("SELECT * FROM tokens;")
    return jsonify(tokens)

@app.route('/add_token', methods=['POST'])
def add_token():
    """🔄 Aggiungi un token via API"""
    data = request.json
    token = data.get('token')
    timestamp = data.get('timestamp', time.time())

    query_database("INSERT INTO tokens (token, timestamp) VALUES (?, ?)", (token, timestamp))
    return jsonify({"message": "Token aggiunto con successo!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

