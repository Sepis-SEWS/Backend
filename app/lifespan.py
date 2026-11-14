"""Model absence degrades readiness, never switches data sources."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    app.state.store.initialize()
    try:
        bundle = app.state.model_loader(app.state.settings.ai_root)
    except (OSError, ImportError, RuntimeError, ValueError, KeyError):
        app.state.inference = None
    else:
        app.state.inference = bundle.state
    try:
        yield
    finally:
        app.state.inference = None
