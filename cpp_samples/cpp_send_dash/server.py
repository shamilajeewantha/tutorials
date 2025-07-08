from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive', methods=['POST'])
def receive():
    data = request.json  # Expecting JSON data
    print("Received POST data:", data)
    return jsonify({"message": "Data received", "your_data": data}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
