import os
from src.ai_court.api.server import app

if __name__ == "__main__":
    # Check for production mode
    if os.environ.get("APP_ENV") == "production":
        try:
            from waitress import serve
            print("Starting production server with Waitress on port 5002...")
            serve(app, host="0.0.0.0", port=5002)
        except ImportError:
            print("Waitress not installed. Falling back to Flask dev server.")
            app.run(debug=False, port=5002, host="0.0.0.0")
    else:
        print("Starting development server...")
        app.run(debug=True, port=5002, host="0.0.0.0")
