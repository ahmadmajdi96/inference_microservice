from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.services.models import load_models

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def startup_event():
    app.state.models = load_models()


app.include_router(router)
