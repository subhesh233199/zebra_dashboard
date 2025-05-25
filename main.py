import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from typing import Dict, List
from models import FolderPathRequest, AnalysisResponse, SharedState
from cache import cleanup_old_cache, hash_string, hash_pdf_contents, get_cached_report, store_cached_report
from pdf_processing import get_pdf_files_from_folder, convert_windows_path
from crew_setup import setup_crew, run_full_analysis
import uvicorn
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RRR Release Analysis Tool", description="API for analyzing release readiness reports")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shared_state = SharedState()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdfs(request: FolderPathRequest):
    try:
        cleanup_old_cache()

        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        folder_path_hash = hash_string(folder_path)
        pdf_files = get_pdf_files_from_folder(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        logger.info(f"Computed hashes - folder_path_hash: {folder_path_hash}, pdfs_hash: {pdfs_hash}")

        cached_response = get_cached_report(folder_path_hash, pdfs_hash)
        if cached_response:
            logger.info(f"Cache hit for folder_path_hash: {folder_path_hash}")
            return cached_response

        logger.info(f"Cache miss for folder_path_hash: {folder_path_hash}, running full analysis")
        response = await run_full_analysis(request)

        store_cached_report(folder_path_hash, pdfs_hash, response)
        return response

    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
