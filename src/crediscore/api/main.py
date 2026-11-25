from fastapi import FastAPI

app = FastAPI(
    title="CrediScore API",
    description="Credit scoring prediction API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
