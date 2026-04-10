"""
Server entry point for OpenEnv multi-mode deployment.
Imports the FastAPI app from main.py and starts uvicorn.
"""
import uvicorn
from main import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
