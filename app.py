from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>NetProphet is LIVE!</h1><p>System status: Online</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)