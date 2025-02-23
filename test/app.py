from flask import Flask, render_template, jsonify, request
from llm import Groq_llm

app = Flask(__name__)

groq = Groq_llm()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/message', methods=['POST'])
def get_message():
    data = request.json
    name = data.get('name', 'World')
    new_message = groq.messageChecker(name)
    return jsonify({"message": f"{new_message}"})

if __name__ == '__main__':
    app.run(debug=True)