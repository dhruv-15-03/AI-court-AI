import os
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Render free instances have 512MB RAM. Keep a single light worker to avoid OOM.
# Override via env vars if you scale up.
workers = int(__import__('os').environ.get('GUNICORN_WORKERS', '1'))
threads = int(__import__('os').environ.get('GUNICORN_THREADS', '2'))
worker_class = "gthread"
worker_tmp_dir = "/dev/shm"

# Disable preloading to prevent eager duplication of model memory across workers.
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"
timeout = 60
