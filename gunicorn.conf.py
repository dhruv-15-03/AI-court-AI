import os
import sys

# Add src directory to Python path so 'ai_court' package can be found
sys.path.append(os.path.join(os.getcwd(), 'src'))

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Render free instances have 512MB RAM. Keep a single light worker to avoid OOM.
# Override via env vars if you scale up.
workers = int(os.environ.get('GUNICORN_WORKERS', '1'))
threads = int(os.environ.get('GUNICORN_THREADS', '2'))
worker_class = "gthread"
worker_tmp_dir = "/dev/shm"

# Disable preloading to prevent eager duplication of model memory across workers.
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"
# First inference after a cold start loads the model lazily and can take well over
# 60s on a free-tier instance; a short timeout kills the worker mid-load (502).
# Override via GUNICORN_TIMEOUT if needed.
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))

# Recycle workers after serving this many requests to guard against memory leaks.
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', '1000'))
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', '50'))
