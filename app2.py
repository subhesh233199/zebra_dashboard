import os
import re
import json
import runpy
import base64
import sqlite3
import hashlib
import time
from typing import List, Dict, Tuple, Any, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
import ssl
import warnings
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

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

# Initialize Azure OpenAI
llm = LLM(
    model=f"azure/{os.getenv('DEPLOYMENT_NAME')}",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.3,
    top_p=0.95,
)

# Constants
START_HEADER_PATTERN = 'Release Readiness Critical Metrics (Previous/Current):'
END_HEADER_PATTERN = 'Release Readiness Functional teams Deliverables Checklist:'
EXPECTED_METRICS = [
    "Open ALL RRR Defects", "Open Security Defects", "All Open Defects (T-1)",
    "All Security Open Defects", "Load/Performance", "E2E Test Coverage",
    "Automation Test Coverage", "Unit Test Coverage", "Defect Closure Rate",
    "Regression Issues", "Customer Specific Testing (UAT)"
]
CACHE_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days in seconds

# Pydantic models
class FolderPathRequest(BaseModel):
    folder_path: str

    @validator('folder_path')
    def validate_folder_path(cls, v):
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v

class AnalysisResponse(BaseModel):
    metrics: Dict
    visualizations: List[str]
    report: str
    evaluation: Dict
    hyperlinks: List[Dict]

class MetricItem(BaseModel):
    version: str
    value: Union[float, str]
    status: str
    trend: Union[str, None] = None

# Shared state for thread-safe data sharing
class SharedState:
    def __init__(self):
        self.metrics = None
        self.report_parts = {}
        self.lock = Lock()
        self.visualization_ready = False
        self.viz_lock = Lock()

shared_state = SharedState()

# SQLite database setup
def init_cache_db():
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report_cache (
            folder_path_hash TEXT PRIMARY KEY,
            pdfs_hash TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_cache_db()

def hash_string(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def hash_pdf_contents(pdf_files_with_versions: List[Tuple[str, str]]) -> str:
    hasher = hashlib.md5()
    for pdf_path, version in sorted(pdf_files_with_versions, key=lambda x: x[1]):
        try:
            with open(pdf_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            hasher.update(version.encode('utf-8'))  # Include version in hash
        except Exception as e:
            logger.error(f"Error hashing PDF {pdf_path}: {str(e)}")
            raise
    return hasher.hexdigest()

def get_cached_report(folder_path_hash: str, pdfs_hash: str) -> Union[AnalysisResponse, None]:
    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT report_json, created_at
            FROM report_cache
            WHERE folder_path_hash = ? AND pdfs_hash = ?
        ''', (folder_path_hash, pdfs_hash))
        result = cursor.fetchone()
        conn.close()

        if result:
            report_json, created_at = result
            current_time = int(time.time())
            if current_time - created_at < CACHE_TTL_SECONDS:
                report_dict = json.loads(report_json)
                return AnalysisResponse(**report_dict)
            else:
                with shared_state.lock:
                    conn = sqlite3.connect('cache.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM report_cache WHERE folder_path_hash = ?', (folder_path_hash,))
                    conn.commit()
                    conn.close()
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached report: {str(e)}")
        return None

def store_cached_report(folder_path_hash: str, pdfs_hash: str, response: AnalysisResponse):
    try:
        report_json = json.dumps(response.dict())
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO report_cache (folder_path_hash, pdfs_hash, report_json, created_at)
                VALUES (?, ?, ?, ?)
            ''', (folder_path_hash, pdfs_hash, report_json, current_time))
            conn.commit()
            conn.close()
        logger.info(f"Cached report for folder_path_hash: {folder_path_hash}")
    except Exception as e:
        logger.error(f"Error storing cached report: {str(e)}")

def cleanup_old_cache():
    try:
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM report_cache
                WHERE created_at < ?
            ''', (current_time - CACHE_TTL_SECONDS,))
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
        logger.info(f"Cleaned up old cache entries, deleted {deleted_rows} rows")
    except Exception as e:
        logger.error(f"Error cleaning up old cache entries: {str(e)}")

def get_pdf_files_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    pdf_files = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    version_pattern = r'(\d+\.\d+)'  # Matches versions like "24.10", "29.30"
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file_name)
            match = re.search(version_pattern, file_name)
            if match:
                version = match.group(1)
                pdf_files.append((full_path, version))
            else:
                logger.warning(f"No version found in filename: {file_name}")
                continue
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the folder {folder_path}.")
    
    # Sort by version
    def version_key(version_tuple: Tuple[str, str]) -> tuple:
        major, minor = map(int, version_tuple[1].split('.'))
        return (major, minor)
    return sorted(pdf_files, key=version_key)

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n'
            if not text.strip():
                raise ValueError(f"No text extracted from {pdf_path}")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise

def extract_hyperlinks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    hyperlinks = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        annot_obj = annot.get_object()
                        if annot_obj['/Subtype'] == '/Link' and '/A' in annot_obj:
                            uri = annot_obj['/A']['/URI']
                            text = page.extract_text() or ""
                            context_start = max(0, text.find(uri) - 50)
                            context_end = min(len(text), text.find(uri) + len(uri) + 50)
                            context = text[context_start:context_end].strip()
                            hyperlinks.append({
                                "url": uri,
                                "context": context,
                                "page": page_num,
                                "source_file": os.path.basename(pdf_path)
                            })
    except Exception as e:
        logger.error(f"Error extracting hyperlinks from {pdf_path}: {str(e)}")
    return hyperlinks

def locate_table(text: str, start_header: str, end_header: str) -> str:
    start_index = text.find(start_header)
    end_index = text.find(end_header)
    if start_index == -1:
        raise ValueError(f'Header {start_header} not found in text')
    if end_index == -1:
        raise ValueError(f'Header {end_header} not found in text')
    table_text = text[start_index:end_index].strip()
    if not table_text:
        raise ValueError(f"No metrics table data found between headers")
    return table_text

def evaluate_with_llm_judge(source_text: str, generated_report: str) -> Tuple[int, str]:
    judge_llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=512,
        timeout=None,
    )
    
    prompt = f"""Act as an impartial judge evaluating report quality. You will be given:
1. ORIGINAL SOURCE TEXT (extracted from PDF)
2. GENERATED REPORT (created by AI)

Evaluate based on:
- Data accuracy (50% weight): Does the report correctly reflect the source data?
- Analysis depth (30% weight): Does it provide meaningful insights?
- Clarity (20% weight): Is the presentation clear and professional?

ORIGINAL SOURCE:
{source_text}

GENERATED REPORT:
{generated_report}

INSTRUCTIONS:
1. Provide a score from 0-100
2. Give brief 2-3 sentence evaluation
3. Use EXACTLY this format:
Score: [0-100]
Evaluation: [your evaluation]

Your evaluation:"""
    
    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content
        score_line = next(line for line in response_text.split('\n') if line.startswith('Score:'))
        score = int(score_line.split(':')[1].strip())
        eval_lines = [line for line in response_text.split('\n') if line.startswith('Evaluation:')]
        evaluation = ' '.join(line.split('Evaluation:')[1].strip() for line in eval_lines)
        return score, evaluation
    except Exception as e:
        logger.error(f"Error parsing judge response: {e}\nResponse was:\n{response_text}")
        return 50, "Could not parse evaluation"

def validate_report(report: str) -> bool:
    required_sections = ["# Software Metrics Report", "## Overview", "## Metrics Summary", "## Key Findings", "## Recommendations"]
    return all(section in report for section in required_sections)

def validate_metrics(metrics: Dict[str, Any]) -> bool:
    if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
        logger.warning(f"Invalid metrics structure: {metrics}")
        return False
    missing_metrics = [m for m in EXPECTED_METRICS if m not in metrics['metrics']]
    if missing_metrics:
        logger.warning(f"Missing metrics: {missing_metrics}")
        return False
    for metric, data in metrics['metrics'].items():
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                logger.warning(f"Invalid ATLS/BTLS structure for {metric}: {data}")
                return False
            for sub in ['ATLS', 'BTLS']:
                if not isinstance(data[sub], list) or len(data[sub]) < 1:
                    logger.warning(f"Empty or insufficient {sub} data for {metric}: {data[sub]}")
                    return False
                has_non_zero = False
                for item in data[sub]:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'value', 'status']):
                            logger.warning(f"Missing keys in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.warning(f"Invalid version in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                            logger.warning(f"Invalid value in {sub} item for {metric}: {item}")
                            return False
                        if item_dict['value'] > 0:
                            has_non_zero = True
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.warning(f"Invalid status in {sub} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.warning(f"Invalid trend in {sub} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.warning(f"Invalid item in {sub} for {metric}: {item}, error: {str(e)}")
                        return False
                if not has_non_zero:
                    logger.warning(f"No non-zero values in {sub} for {metric}")
                    return False
        elif metric == "Customer Specific Testing (UAT)":
            if not isinstance(data, dict) or not all(client in data for client in ['RBS', 'Tesco', 'Belk']):
                logger.warning(f"Invalid structure for {metric}: {data}")
                return False
            for client in ['RBS', 'Tesco', 'Belk']:
                client_data = data.get(client, [])
                if not isinstance(client_data, list) or len(client_data) < 1:
                    logger.warning(f"Empty or insufficient data for {metric} {client}: {client_data}")
                    return False
                for item in client_data:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'pass_count', 'fail_count', 'status']):
                            logger.warning(f"Missing keys in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.warning(f"Invalid version in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['pass_count'], (int, float)) or item_dict['pass_count'] < 0:
                            logger.warning(f"Invalid pass_count in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['fail_count'], (int, float)) or item_dict['fail_count'] < 0:
                            logger.warning(f"Invalid fail_count in {client} item for {metric}: {item}")
                            return False
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.warning(f"Invalid status in {client} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.warning(f"Invalid trend in {client} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.warning(f"Invalid item in {client} for {metric}: {item}, error: {str(e)}")
                        return False
        else:  # Non-ATLS/BTLS metrics
            if not isinstance(data, list) or len(data) < 1:
                logger.warning(f"Empty or insufficient data for {metric}: {data}")
                return False
            has_non_zero = False
            for item in data:
                try:
                    item_dict = dict(item)
                    if not all(k in item_dict for k in ['version', 'value', 'status']):
                        logger.WARNING(f"Missing keys in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                        logger.warning(f"Invalid version in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                        logger.warning(f"Invalid value in item for {metric}: {item}")
                        return False
                    if item_dict['value'] > 0:
                        has_non_zero = True
                    if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                        logger.warning(f"Invalid status in item for {metric}: {item}")
                        return False
                    if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                        logger.warning(f"Invalid trend in item for {metric}: {item}")
                        return False
                exceptijdensException as e:
                    logger.warning(f"Invalid item for {metric}: {item}, error: {str(e)}")
                    return False
            if not has_non_zero:
                logger.warning(f"No non-zero values for {metric}")
                return False
    return True

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def process_task_output(raw_output: str) -> Dict:
    logger.info(f"Processing task output: {raw_output[:200]}...")
    data = clean_json_output(raw_output)
    if not validate_metrics(data):
        logger.error(f"Validation failed for processed output: {json.dumps(data, indent=2)[:200]}...")
        raise ValueError("Invalid or incomplete metrics data")
    # Validate and correct trends
    for metric, metric_data in data['metrics'].items():
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            for sub in ['ATLS', 'BTLS']:
                items = sorted(metric_data[sub], key=lambda x: tuple(map(int, x['version'].split('.'))))
                for i in range(len(items)):
                    if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                        items[i]['trend'] = '→'
                    else:
                        prev_val = float(items[i-1]['value'])
                        curr_val = float(items[i]['value'])
                        if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = ((curr_val - prev_val) / prev_val) * 100
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        elif metric == "Customer Specific Testing (UAT)":
            for client in ['RBS', 'Tesco', 'Belk']:
                items = sorted(metric_data[client], key=lambda x: tuple(map(int, x['version'].split('.'))))
                for i in range(len(items)):
                    pass_count = float(items[i].get('pass_count', 0))
                    fail_count = float(items[i].get('fail_count', 0))
                    total = pass_count + fail_count
                    pass_rate = (pass_count / total * 100) if total > 0 else 0
                    items[i]['pass_rate'] = pass_rate
                    if i == 0:
                        items[i]['trend'] = '→'
                    else:
                        prev_pass_count = float(items[i-1].get('pass_count', 0))
                        prev_fail_count = float(items[i-1].get('fail_count', 0))
                        prev_total = prev_pass_count + prev_fail_count
                        prev_pass_rate = (prev_pass_count / prev_total * 100) if prev_total > 0 else 0
                        if prev_total == 0 or total == 0 or abs(pass_rate - prev_pass_rate) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = pass_rate - prev_pass_rate
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        else:  # Non-ATLS/BTLS metrics
            items = sorted(metric_data, key=lambda x: tuple(map(int, x['version'].split('.'))))
            for i in range(len(items)):
                if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                    items[i]['trend'] = '→'
                else:
                    prev_val = float(items[i-1]['value'])
                    curr_val = float(items[i]['value'])
                    if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                        items[i]['trend'] = '→'
                    else:
                        pct_change = ((curr_val - prev_val) / prev_val) * 100
                        if abs(pct_change) < 1:
                            items[i]['trend'] = '→'
                        elif pct_change > 0:
                            items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                        else:
                            items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
    return data

def setup_crew(extracted_text: str, versions: List[str], llm=llm) -> tuple:
    if not versions:
        raise ValueError("No versions provided for crew setup")
    
    structurer = Agent(
        role="Data Architect",
        goal="Structure raw release data into VALID JSON format",
        backstory="Expert in transforming unstructured data into clean JSON structures",
        llm=llm,
        verbose=True,
        memory=True,
    )

    validated_structure_task = Task(
        description=f"""Convert this release data to STRICT JSON:
{extracted_text}

RULES:
1. Output MUST be valid JSON only
2. Use this EXACT structure:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{"ATLS": [{{"version": "{versions[0]}", "value": N, "status": "TEXT"}}, ...], "BTLS": [...]}},
        "Open Security Defects": {{"ATLS": [...], "BTLS": [...]}},
        "All Open Defects (T-1)": {{"ATLS": [...], "BTLS": [...]}},
        "All Security Open Defects": {{"ATLS": [...], "BTLS": [...]}},
        "Load/Performance": {{"ATLS": [...], "BTLS": [...]}},
        "E2E Test Coverage": [{{"version": "{versions[0]}", "value": N, "status": "TEXT"}}, ...],
        "Automation Test Coverage": [...],
        "Unit Test Coverage": [...],
        "Defect Closure Rate": [...],
        "Regression Issues": [...],
        "Customer Specific Testing (UAT)": {{
            "RBS": [{{"version": "{versions[0]}", "pass_count": N, "fail_count": M, "status": "TEXT"}}, ...],
            "Tesco": [...],
            "Belk": [...]
        }}
    }}
}}
3. Include ALL metrics: {', '.join(EXPECTED_METRICS)}
4. Use versions: {', '.join(versions)}
5. For UAT, pass_count and fail_count must be non-negative integers, at least one non-zero per client
6. For other metrics, values must be positive numbers (at least one non-zero per metric)
7. Status must be one of: "ON TRACK", "MEDIUM RISK", "RISK", "NEEDS REVIEW"
8. Ensure at least 1 item per metric/sub-metric, matching provided versions
9. No text outside JSON, no trailing commas, no comments
10. Validate JSON syntax before output
EXAMPLE:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{
            "ATLS": [
                {{"version": "{versions[0]}", "value": 10, "status": "RISK"}},
                {{"version": "{versions[1]}" if len(versions) > 1 else versions[0], "value": 8, "status": "MEDIUM RISK"}}
                {',' + '{"version": "' + versions[2] + '", "value": 5, "status": "ON TRACK"}}' if len(versions) > 2 else ''}
            ],
            "BTLS": [...]
        }},
        "Customer Specific Testing (UAT)": {{
            "RBS": [
                {{"version": "{versions[0]}", "pass_count": 50, "fail_count": 5, "status": "ON TRACK"}},
                {{"version": "{versions[1]}" if len(versions) > 1 else versions[0], "pass_count": 48, "fail_count": 6, "status": "MEDIUM RISK"}}
                {',' + '{"version": "' + versions[2] + '", "pass_count": 52, "fail_count": 4, "status": "ON TRACK"}}' if len(versions) > 2 else ''}
            ],
            "Tesco": [...],
            "Belk": [...]
        }},
        ...
    }}
}}""",
        agent=structurer,
        async_execution=False,
        expected_output="Valid JSON string with no extra text",
        callback=lambda output: (
            logger.info(f"Structure task output: {output.raw[:200]}..."),
            setattr(shared_state, 'metrics', process_task_output(output.raw))
        )
    )

    analyst = Agent(
        role="Trend Analyst",
        goal="Add accurate trends to metrics data and maintain valid JSON",
        backstory="Data scientist specializing in metric analysis",
        llm=llm,
        verbose=True,
        memory=True,
    )

    analysis_task = Task(
        description=f"""Enhance metrics JSON with trends:
1. Input is JSON from Data Structurer
2. Add 'trend' field to each metric item
3. Output MUST be valid JSON
4. For metrics except Customer Specific Testing (UAT):
   - Sort items by version ({', '.join(versions)})
   - For each item (except first per metric):
     - Compute % change: ((current_value - previous_value) / previous_value) * 100
     - If previous_value is 0 or |change| < 0.01, set trend to "→"
     - If |% change| < 1%, set trend to "→"
     - If % change > 0, set trend to "↑ (X.X%)"
     - If % change < 0, set trend to "↓ (X.X%)"
   - First item per metric gets "→"
5. For Customer Specific Testing (UAT):
   - For each client (RBS, Tesco, Belk), compute pass rate: pass_count / (pass_count + fail_count) * 100
   - Sort items by version ({', '.join(versions)})
   - For each item (except first per client):
     - Compute % change in pass rate: (current_pass_rate - previous_pass_rate)
     - If previous_total or current_total is 0 or |change| < 0.01, set trend to "→"
     - If |% change| < 1%, set trend to "→"
     - If % change > 0, set trend to "↑ (X.X%)"
     - If % change < 0, set trend to "↓ (X.X%)"
   - First item per client gets "→"
6. Ensure all metrics are included: {', '.join(EXPECTED_METRICS)}
7. Use double quotes for all strings
8. No trailing commas or comments
9. Validate JSON syntax before output
EXAMPLE OUTPUT:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{
            "ATLS": [
                {{"version": "{versions[0]}", "value": 10, "status": "RISK", "trend": "→"}},
                {{"version": "{versions[1]}" if len(versions) > 1 else versions[0], "value": 8, "status": "MEDIUM RISK", "trend": "↓ (20.0%)"}}
                {',' + '{"version": "' + versions[2] + '", "value": 5, "status": "ON TRACK", "trend": "↓ (37.5%)"}}' if len(versions) > 2 else ''}
            ],
            "BTLS": [...]
        }},
        "Customer Specific Testing (UAT)": {{
            "RBS": [
                {{"version": "{versions[0]}", "pass_count": 50, "fail_count": 5, "status": "ON TRACK", "pass_rate": 90.9, "trend": "→"}},
                {{"version": "{versions[1]}" if len(versions) > 1 else versions[0], "pass_count": 48, "fail_count": 6, "status": "MEDIUM RISK", "pass_rate": 88.9, "trend": "↓ (2.0%)"}}
                {',' + '{"version": "' + versions[2] + '", "pass_count": 52, "fail_count": 4, "status": "ON TRACK", "pass_rate": 92.9, "trend": "↑ (4.0%)"}}' if len(versions) > 2 else ''}
            ],
            "Tesco": [...],
            "Belk": [...]
        }},
        ...
    }}
}}""",
        agent=analyst,
        async_execution=True,
        context=[validated_structure_task],
        expected_output="Valid JSON string with trend analysis",
        callback=lambda output: (
            logger.info(f"Analysis task output: {output.raw[:200]}..."),
            setattr(shared_state, 'metrics', process_task_output(output.raw))
        )
    )

    visualizer = Agent(
        role="Data Visualizer",
        goal="Generate consistent visualizations for all metrics",
        backstory="Expert in generating Python plots for software metrics",
        llm=llm,
        verbose=True,
        memory=True,
    )

    visualization_task = Task(
        description=f"""Create a standalone Python script that:
1. Accepts the provided 'metrics' JSON structure as input.
2. Generates exactly 10 visualizations for the following metrics, using the specified chart types:
   - Open ALL RRR Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - Open Security Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - All Open Defects (T-1) (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - All Security Open Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - Load/Performance (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - E2E Test Coverage: Line chart showing trend across releases.
   - Automation Test Coverage: Line chart showing trend across releases.
   - Unit Test Coverage: Line chart showing trend across releases.
   - Defect Closure Rate (ATLS): Bar chart showing values across releases.
   - Regression Issues: Bar chart showing values across releases.
3. If Pass/Fail metrics are present in the JSON, generate additional grouped bar charts comparing Pass vs. Fail counts across releases.
4. Each plot must use: plt.figure(figsize=(8,5), dpi=120).
5. Save each chart as a PNG in 'visualizations/' directory with descriptive filenames (e.g., 'open_rrr_defects_atls_btls.png', 'e2e_test_coverage.png').
6. Include error handling for missing or malformed data, ensuring all specified charts are generated.
7. Log each chart generation attempt to 'visualization.log' for debugging.
8. Output ONLY the Python code, with no markdown or explanation text.
9. Do not generate charts for Delivery Against Requirements or Customer Specific Testing (RBS, Tesco, Belk).
10. Ensure exactly 10 charts are generated for the listed metrics, plus additional charts for Pass/Fail metrics if present.
11. For grouped bar charts, use distinct colors for ATLS and BTLS (e.g., blue for ATLS, orange for BTLS) and include a legend.
12. Use versions: {', '.join(versions)}
13. Use the following metric lists for iteration:
    atls_btls_metrics = {EXPECTED_METRICS[:5]}
    coverage_metrics = {EXPECTED_METRICS[5:8]}
    other_metrics = {EXPECTED_METRICS[8:10]}
    Do not use a variable named 'expected_metrics'.""",
        agent=visualizer,
        context=[analysis_task],
        expected_output="Python script only"
    )

    reporter = Agent(
        role="Technical Writer",
        goal="Generate a professional markdown report",
        backstory="Writes structured software metrics reports",
        llm=llm,
        verbose=True,
        memory=True,
    )

    overview_task = Task(
        description=f"""Write ONLY the following Markdown section:
## Overview
- Provide a 3-4 sentence comprehensive summary of release health, covering overall stability, notable improvements, and any concerning patterns observed across releases {', '.join(versions)}
- Explicitly list all analyzed releases
- Include 2-3 notable metric highlights with specific version comparisons where relevant
- Mention any significant deviations from expected patterns
Only output this section.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown for Overview section"
    )

    metrics_summary_task = Task(
        description=f"""Write ONLY the '## Metrics Summary' section with the following order:
### Delivery Against Requirements  
### Open ALL RRR Defects (ATLS)  
### Open ALL RRR Defects (BTLS)  
### Open Security Defects (ATLS)  
### Open Security Defects (BTLS)  
### All Open Defects (T-1) (ATLS)  
### All Open Defects (T-1) (BTLS)  
### All Security Open Defects (ATLS)  
### All Security Open Defects (BTLS)  
### Customer Specific Testing (UAT)  
#### RBS  
#### Tesco  
#### Belk  
### Load/Performance  
#### ATLS  
#### BTLS
### E2E Test Coverage  
### Automation Test Coverage  
### Unit Test Coverage  
### Defect Closure Rate (ATLS)  
### Regression Issues  

STRICT RULES:
- For Customer Specific Testing (UAT), generate tables for each client with the following columns: Release | Pass Count | Fail Count | Pass Rate (%) | Trend | Status
- For other metrics, use existing table formats
- Use only these statuses: ON TRACK, MEDIUM RISK, RISK, NEEDS REVIEW
- Use only these trend formats: ↑ (X%), ↓ (Y%), →
- Use versions: {', '.join(versions)}
- No extra formatting
EXAMPLE FOR UAT:
#### RBS
| Release | Pass Count | Fail Count | Pass Rate (%) | Trend      | Status       |
|---------|------------|------------|---------------|------------|--------------|
| {versions[0]} | 50         | 5          | 90.9          | →          | ON TRACK     |
| {versions[1] if len(versions) > 1 else versions[0]} | 48         | 6          | 88.9          | ↓ (2.0%)   | MEDIUM RISK  |
{('| ' + versions[2] + ' | 52 | 4 | 92.9 | ↑ (4.0%) | ON TRACK |') if len(versions) > 2 else ''}
Only output this section.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Markdown for Metrics Summary"
    )

    key_findings_task = Task(
        description=f"""Generate ONLY this Markdown section:
## Key Findings
1. First finding (2-3 sentences explaining the observation with specific metric references and version comparisons)
2. Second finding (2-3 sentences with quantitative data points from the metrics where relevant)
3. Third finding (2-3 sentences focusing on security-related observations)
4. Fourth finding (2-3 sentences about testing coverage trends)
5. Fifth finding (2-3 sentences highlighting any unexpected patterns or anomalies)
6. Sixth finding (2-3 sentences about performance or load metrics)
7. Seventh finding (2-3 sentences summarizing defect management effectiveness)

Maintain professional, analytical tone while being specific.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    recommendations_task = Task(
        description=f"""Generate ONLY this Markdown section:
## Recommendations
1. First recommendation (2-3 actionable sentences with specific metrics or areas to address)
2. Second recommendation (2-3 sentences about security improvements with version targets)
3. Third recommendation (2-3 sentences about testing coverage enhancements)
4. Fourth recommendation (2-3 sentences about defect management process changes)
5. Fifth recommendation (2-3 sentences about performance optimization)
6. Sixth recommendation (2-3 sentences about risk mitigation strategies)
7. Seventh recommendation (2-3 sentences about monitoring improvements)

Each recommendation should be specific, measurable, and tied to the findings.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    assemble_report_task = Task(
        description="""Assemble the final markdown report in this exact structure:

# Software Metrics Report  

## Overview  
[Insert from Overview Task]  

---  

## Metrics Summary  
[Insert from Metrics Summary Task]  

---  

## Key Findings  
[Insert from Key Findings Task]  

---  

## Recommendations  
[Insert from Recommendations Task]

Do NOT alter content. Just combine with correct formatting.""",
        agent=reporter,
        context=[
            overview_task,
            metrics_summary_task,
            key_findings_task,
            recommendations_task
        ],
        expected_output="Full markdown report"
    )

    data_crew = Crew(
        agents=[structurer, analyst],
        tasks=[validated_structure_task, analysis_task],
        process=Process.sequential,
        verbose=True
    )

    report_crew = Crew(
        agents=[reporter],
        tasks=[overview_task, metrics_summary_task, key_findings_task, recommendations_task, assemble_report_task],
        process=Process.sequential,
        verbose=True
    )

    viz_crew = Crew(
        agents=[visualizer],
        tasks=[visualization_task],
        process=Process.sequential,
        verbose=True
    )

    for crew, name in [(data_crew, "data_crew"), (report_crew, "report_crew"), (viz_crew, "viz_crew")]:
        for i, task in enumerate(crew.tasks):
            if not isinstance(task, Task):
                logger.error(f"Invalid task in {name} at index {i}: {task}")
                raise ValueError(f"Task in {name} is not a Task object")
            logger.info(f"{name} task {i} async_execution: {task.async_execution}")

    return data_crew, report_crew, viz_crew

def clean_json_output(raw_output: str, versions: List[str]) -> dict:
    logger.info(f"Raw analysis output: {raw_output[:200]}...")
    
    # Ensure at least one version for fallback
    fallback_versions = versions if versions else ["unknown.1"]
    if len(fallback_versions) < 2:
        fallback_versions.append(f"{fallback_versions[0].split('.')[0]}.{int(fallback_versions[0].split('.')[1]) + 1}")
    
    # Synthetic data with all values set to 0
    default_json = {
        "metrics": {
            metric: {
                "ATLS": [
                    {"version": fallback_versions[0], "value": 0, "status": "NEEDS REVIEW"},
                    {"version": fallback_versions[1], "value": 0, "status": "NEEDS REVIEW"}
                ],
                "BTLS": [
                    {"version": fallback_versions[0], "value": 0, "status": "NEEDS REVIEW"},
                    {"version": fallback_versions[1], "value": 0, "status": "NEEDS REVIEW"}
                ]
            } if metric in EXPECTED_METRICS[:5] else
            {
                "RBS": [
                    {"version": fallback_versions[0], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"},
                    {"version": fallback_versions[1], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                ],
                "Tesco": [
                    {"version": fallback_versions[0], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"},
                    {"version": fallback_versions[1], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                ],
                "Belk": [
                    {"version": fallback_versions[0], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"},
                    {"version": fallback_versions[1], "pass_count": 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                ]
            } if metric == "Customer Specific Testing (UAT)" else
            [
                {"version": fallback_versions[0], "value": 0, "status": "NEEDS REVIEW"},
                {"version": fallback_versions[1], "value": 0, "status": "NEEDS REVIEW"}
            ]
            for i, metric in enumerate(EXPECTED_METRICS)
        }
    }

    try:
        data = json.loads(raw_output)
        if validate_metrics(data, is_fallback=False):
            return data
        logger.warning(f"Direct JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output, re.MULTILINE)
        if cleaned:
            data = json.loads(cleaned.group(1))
            if validate_metrics(data, is_fallback=False):
                return data
            logger.warning(f"Code block JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Code block JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'\{[\s\S]*\}', raw_output, re.MULTILINE)
        if cleaned:
            json_str = cleaned.group(0)
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            data = json.loads(json_str)
            if validate_metrics(data, is_fallback=False):
                return data
            logger.warning(f"JSON-like structure invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON-like structure parsing failed: {str(e)}")

    logger.error(f"Failed to parse JSON, using default structure with zero values for versions: {fallback_versions}")
    return default_json
def enhance_report_markdown(md_text):
    cleaned = re.sub(r'^```markdown\n|\n```$', '', md_text, flags=re.MULTILINE)
    
    cleaned = re.sub(
        r'(\|.+\|)\n\s*(\|-+\|)',
        r'\1\n\2',
        cleaned
    )
    
    cleaned = re.sub(r'^#\s+(.+)$', r'# \1\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^##\s+(.+)$', r'## \1\n', cleaned, flags=re.MULTILINE)
    
    status_map = {
        "MEDIUM RISK": "**MEDIUM RISK**",
        "HIGH RISK": "**HIGH RISK**",
        "LOW RISK": "**LOW RISK**",
        "ON TRACK": "**ON TRACK**"
    }
    for k, v in status_map.items():
        cleaned = cleaned.replace(k, v)
    
    cleaned = re.sub(r'^\s*-\s+(.+)', r'- \1', cleaned, flags=re.MULTILINE)
    
    return cleaned.encode('utf-8').decode('utf-8')

def convert_windows_path(path: str) -> str:
    path = path.replace('\\', '/')
    path = path.replace('//', '/')
    return path

def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {str(e)}")
        return ""

def run_fallback_visualization(metrics: Dict[str, Any], versions: List[str]):
    with shared_state.viz_lock:
        try:
            os.makedirs("visualizations", exist_ok=True)
            logging.basicConfig(level=logging.INFO, filename='visualization.log')
            logger = logging.getLogger(__name__)
            logger.info("Starting fallback visualization")

            if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
                logger.error(f"Invalid metrics data: {metrics}")
                raise ValueError("Metrics data is empty or invalid")

            atls_btls_metrics = EXPECTED_METRICS[:5]
            coverage_metrics = EXPECTED_METRICS[5:8]
            other_metrics = EXPECTED_METRICS[8:10]

            generated_files = []
            for metric in atls_btls_metrics:
                try:
                    data = metrics['metrics'].get(metric, {})
                    if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                        logger.warning(f"Creating placeholder for {metric}: invalid or missing ATLS/BTLS data")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    atls_data = data.get('ATLS', [])
                    btls_data = data.get('BTLS', [])
                    version_list = versions
                    atls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in atls_data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                    btls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in btls_data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                    if not version_list or len(atls_values) != len(version_list) or len(btls_values) != len(version_list):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    x = np.arange(len(version_list))
                    width = 0.35
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(x - width/2, atls_values, width, label='ATLS', color='blue')
                    plt.bar(x + width/2, btls_values, width, label='BTLS', color='orange')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    plt.xticks(x, version_list)
                    plt.legend()
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated grouped bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            for metric in coverage_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    version_list = versions
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                    if not version_list or len(values) != len(version_list):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.plot(version_list, values, marker='o', color='green')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated line chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            for metric in other_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    version_list = versions
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                    if not version_list or len(values) != len(version_list):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(version_list, values, color='purple')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            if 'Pass/Fail' in metrics['metrics']:
                try:
                    data = metrics['metrics'].get('Pass/Fail', {})
                    if not isinstance(data, dict):
                        logger.warning(f"Creating placeholder for Pass/Fail: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, "No data for Pass/Fail", ha='center', va='center')
                        plt.title("Pass/Fail Metrics")
                        filename = 'visualizations/pass_fail.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for Pass/Fail: {filename}")
                    else:
                        pass_data = data.get('Pass', [])
                        fail_data = data.get('Fail', [])
                        version_list = versions
                        pass_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in pass_data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                        fail_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in fail_data if isinstance(item, dict) and 'version' in item and item['version'] in versions]
                        if not version_list or len(pass_values) != len(version_list) or len(fail_values) != len(version_list):
                            logger.warning(f"Creating placeholder for Pass/Fail: inconsistent data lengths")
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.text(0.5, 0.5, "Incomplete data for Pass/Fail", ha='center', va='center')
                            plt.title("Pass/Fail Metrics")
                            filename = 'visualizations/pass_fail.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated placeholder chart for Pass/Fail: {filename}")
                        else:
                            x = np.arange(len(version_list))
                            width = 0.35
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.bar(x - width/2, pass_values, width, label='Pass', color='green')
                            plt.bar(x + width/2, fail_values, width, label='Fail', color='red')
                            plt.xlabel('Release')
                            plt.ylabel('Count')
                            plt.title('Pass/Fail Metrics')
                            plt.xticks(x, version_list)
                            plt.legend()
                            filename = 'visualizations/pass_fail.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated grouped bar chart for Pass/Fail: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for Pass/Fail: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, "Error generating Pass/Fail", ha='center', va='center')
                    plt.title("Pass/Fail Metrics")
                    filename = 'visualizations/pass_fail.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for Pass/Fail: {filename}")

            logger.info(f"Completed fallback visualization, generated {len(generated_files)} files")
        except Exception as e:
            logger.error(f"Fallback visualization failed: {str(e)}")
            raise
        finally:
            plt.close('all')

async def run_full_analysis(request: FolderPathRequest) -> AnalysisResponse:
    folder_path = convert_windows_path(request.folder_path)
    folder_path = os.path.normpath(folder_path)
    
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path does not exist: {folder_path}")
    
    pdf_files_with_versions = get_pdf_files_from_folder(folder_path)
    pdf_files = [path for path, _ in pdf_files_with_versions]
    versions = [version for _, version in pdf_files_with_versions]
    if not versions:
        raise HTTPException(status_code=400, detail="No valid versions extracted from PDF filenames")
    logger.info(f"Processing {len(pdf_files)} PDF files with versions: {versions}")

    # Parallel PDF processing
    extracted_texts = []
    all_hyperlinks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        text_futures = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in pdf_files}
        hyperlink_futures = {executor.submit(extract_hyperlinks_from_pdf, pdf): pdf for pdf in pdf_files}
        
        for future in as_completed(text_futures):
            pdf = text_futures[future]
            try:
                text = locate_table(future.result(), START_HEADER_PATTERN, END_HEADER_PATTERN)
                extracted_texts.append((os.path.basename(pdf), text))
            except Exception as e:
                logger.error(f"Failed to process text from {pdf}: {str(e)}")
                continue
        
        for future in as_completed(hyperlink_futures):
            pdf = hyperlink_futures[future]
            try:
                all_hyperlinks.extend(future.result())
            except Exception as e:
                logger.error(f"Failed to process hyperlinks from {pdf}: {str(e)}")
                continue

    if not extracted_texts:
        raise HTTPException(status_code=400, detail="No valid text extracted from PDFs")

    full_source_text = "\n".join(
        f"File: {name}\n{text}" for name, text in extracted_texts
    )

    # Pass versions to setup_crew
    data_crew, report_crew, viz_crew = setup_crew(full_source_text, versions, llm)
    
    logger.info("Starting data_crew")
    await data_crew.kickoff_async()
    logger.info("Data_crew completed")
    
    for i, task in enumerate(data_crew.tasks):
        if not hasattr(task, 'output') or not hasattr(task.output, 'raw'):
            logger.error(f"Invalid output for data_crew task {i}: {task}")
            raise ValueError(f"Data crew task {i} did not produce a valid output")
        logger.info(f"Data_crew task {i} output: {task.output.raw[:200]}...")

    if not shared_state.metrics or not isinstance(shared_state.metrics, dict):
        logger.error(f"Invalid metrics in shared_state: type={type(shared_state.metrics)}, value={shared_state.metrics}")
        raise HTTPException(status_code=500, detail="Failed to generate valid metrics data")
    logger.info(f"Metrics after data_crew: {json.dumps(shared_state.metrics, indent=2)[:200]}...")

    logger.info("Starting report_crew and viz_crew")
    await asyncio.gather(
        report_crew.kickoff_async(),
        viz_crew.kickoff_async()
    )
    logger.info("Report_crew and viz_crew completed")

    if not hasattr(report_crew.tasks[-1], 'output') or not hasattr(report_crew.tasks[-1].output, 'raw'):
        logger.error(f"Invalid output for report_crew task {report_crew.tasks[-1]}")
        raise ValueError("Report crew did not produce a valid output")
    logger.info(f"Report_crew output: {report_crew.tasks[-1].output.raw[:100]}...")

    if not hasattr(viz_crew.tasks[0], 'output') or not hasattr(viz_crew.tasks[0].output, 'raw'):
        logger.error(f"Invalid output for viz_crew task {viz_crew.tasks[0]}")
        raise ValueError("Visualization crew did not produce a valid output")
    logger.info(f"Viz_crew output: {viz_crew.tasks[0].output.raw[:100]}...")

    metrics = shared_state.metrics

    enhanced_report = enhance_report_markdown(report_crew.tasks[-1].output.raw)
    if not validate_report(enhanced_report):
        logger.error("Report missing required sections")
        raise HTTPException(status_code=500, detail="Generated report is incomplete")

    viz_folder = "visualizations"
    if os.path.exists(viz_folder):
        shutil.rmtree(viz_folder)
    os.makedirs(viz_folder, exist_ok=True)

    script_path = "visualizations.py"
    raw_script = viz_crew.tasks[0].output.raw
    clean_script = re.sub(r'```python|```$', '', raw_script, flags=re.MULTILINE).strip()

    try:
        with shared_state.viz_lock:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(clean_script)
            logger.info(f"Visualization script written to {script_path}")
            logger.debug(f"Visualization script content:\n{clean_script}")
            runpy.run_path(script_path, init_globals={'metrics': metrics})
            logger.info("Visualization script executed successfully")
    except Exception as e:
        logger.error(f"Visualization script failed: {str(e)}")
        logger.info("Running fallback visualization")
        run_fallback_visualization(metrics, versions)

    viz_base64 = []
    expected_count = 10 + (1 if 'Pass/Fail' in metrics.get('metrics', {}) else 0)
    min_visualizations = 5
    if os.path.exists(viz_folder):
        viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
        for img in viz_files:
            img_path = os.path.join(viz_folder, img)
            base64_str = get_base64_image(img_path)
            if base64_str:
                viz_base64.append(base64_str)
        logger.info(f"Generated {len(viz_base64)} visualizations, expected {expected_count}, minimum required {min_visualizations}")
        if len(viz_base64) < min_visualizations:
            logger.warning("Insufficient visualizations, running fallback")
            run_fallback_visualization(metrics, versions)
            viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            viz_base64 = []
            for img in viz_files:
                img_path = os.path.join(viz_folder, img)
                base64_str = get_base64_image(img_path)
                if base64_str:
                    viz_base64.append(base64_str)
            if len(viz_base64) < min_visualizations:
                logger.error(f"Still too few visualizations: {len(viz_base64)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate minimum required visualizations: got {len(viz_base64)}, need at least {min_visualizations}"
                )

    score, evaluation = evaluate_with_llm_judge(full_source_text, enhanced_report)

    return AnalysisResponse(
        metrics=metrics,
        visualizations=viz_base64,
        report=enhanced_report,
        evaluation={"score": score, "text": evaluation},
        hyperlinks=all_hyperlinks
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdfs(request: FolderPathRequest):
    try:
        cleanup_old_cache()

        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        folder_path_hash = hash_string(folder_path)
        pdf_files_with_versions = get_pdf_files_from_folder(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files_with_versions)
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
    finally:
        plt.close('all')

app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
