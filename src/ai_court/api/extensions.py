from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from ai_court.api import config

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60 per minute"],
    storage_uri=config.RATE_LIMIT_STORAGE_URI,
)
