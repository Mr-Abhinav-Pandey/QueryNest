import os

from dotenv import load_dotenv

# Load environment variables once at import time so the rest of the backend
# can rely on plain constants.
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

# Validate required runtime configuration
_REQUIRED_ENV_VARS = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_KEY,
    "SUPABASE_BUCKET": SUPABASE_BUCKET,
}

missing_vars = [
    name for name, value in _REQUIRED_ENV_VARS.items()
    if not value
]

if missing_vars:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing_vars)
        + ". Please update backend/.env before starting QueryNest."
    )

# Retrieval configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
SIGNED_URL_TTL_SECONDS = 3600

# Operational metadata
APPLICATION_NAME = "QueryNest"
APPLICATION_VERSION = "1.0.0-dev"
APP_ENVIRONMENT = os.getenv("QUERYNEST_ENV", "development")