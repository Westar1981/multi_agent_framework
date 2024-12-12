"""Main FastAPI application for the web interface."""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import pointcut_routes

# Create FastAPI app
app = FastAPI(title="Multi-Agent Framework Dashboard")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Include routers
app.include_router(pointcut_routes.router)

# Root route
@app.get("/")
async def root():
    """Redirect to dashboard."""
    return {"message": "Welcome to Multi-Agent Framework Dashboard"} 