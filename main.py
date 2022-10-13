import uvicorn
import os
from app import app

defined_port = int(os.getenv("PORT")) if os.getenv("PORT") else None

def main():
    uvicorn.run(app, host="0.0.0.0", port=defined_port or 8000)


if __name__ == "__main__":
    main()
