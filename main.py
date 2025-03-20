from fastapi import FastAPI
from api.routes.proposal import router as proposal_router
from api.routes.estimation import router as estimation_router
from api.routes.response_suggester import router as response_suggester_router

app = FastAPI(
    title="Freelancing Automation API",
    description="API for generating job proposals and estimating project time, cost, and resources",
    version="1.0.0"
)

# Include routers
app.include_router(proposal_router)
app.include_router(estimation_router)
app.include_router(response_suggester_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)