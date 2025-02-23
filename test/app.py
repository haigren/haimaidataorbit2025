from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/message', methods=['POST'])
def get_message():
    data = request.json
    name = data.get('name', 'World')
    return jsonify({"message": f"Hello, {name}! Welcome to your Python-powered web app 🚀"})

if __name__ == '__main__':
    app.run(debug=True)
