import os
from src.ai_court.api.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    # Check for production mode
    if os.environ.get("APP_ENV") == "production":
        try:
            from waitress import serve
            print(f"Starting production server with Waitress on port {port}...")
            serve(app, host="0.0.0.0", port=port)
        except ImportError:
            print("Waitress not installed. Falling back to Flask dev server.")
            app.run(debug=False, port=port, host="0.0.0.0")
    else:
        print(f"Starting development server on port {port}...")
        app.run(debug=True, port=port, host="0.0.0.0")
