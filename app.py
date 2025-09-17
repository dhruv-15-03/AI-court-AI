from src.ai_court.api.server import app

if __name__ == "__main__":
    app.run(debug=True, port=5002, host='0.0.0.0')