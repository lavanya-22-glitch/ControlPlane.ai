import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from controlplane.policy.registry import policy_registry
from controlplane.telemetry.sink import telemetry_sink
from controlplane.proxy.router import router
from controlplane.proxy.dispatcher import dispatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("controlplane")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: load policy registry & initialize database
    config_path = os.getenv("CONTROLPLANE_POLICY_PATH", "config/policies.yaml")
    policy_registry._config_path = Path(config_path)
    policy_registry.reload()

    db_path = os.getenv("CONTROLPLANE_DB_PATH", "data/audit_logs.db")
    telemetry_sink.db_path = Path(db_path)
    await telemetry_sink.initialize()

    logger.info("ControlPlane.ai Proxy initialized successfully.")
    yield

    # 2. Shutdown: cleanup dispatcher connections
    await dispatcher.close()
    logger.info("ControlPlane.ai Proxy shutdown complete.")


app = FastAPI(
    title="ControlPlane.ai",
    description="High-performance, drop-in reverse proxy for LLM safety, governance & auditing.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web dashboard & frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router)

# Mount Dashboard Static Files if present
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("controlplane.main:app", host="0.0.0.0", port=port, reload=True)
