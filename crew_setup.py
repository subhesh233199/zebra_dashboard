from crewai import Agent, Task, Crew, Process
from models import SharedState, EXPECTED_METRICS
from data_processing import clean_json_output, process_task_output
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def setup_crew(extracted_text: str, llm) -> tuple:
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
[... rest of the task description ...]""",
        agent=structurer,
        async_execution=False,
        expected_output="Valid JSON string with no extra text",
        callback=lambda output: (
            logger.info(f"Structure task output: {output.raw[:200]}..."),
            setattr(shared_state, 'metrics', process_task_output(output.raw))
        )
    )

    [... rest of the crew setup code ...]

    return data_crew, report_crew, viz_crew

async def run_full_analysis(request: FolderPathRequest) -> AnalysisResponse:
    [... rest of the analysis orchestration code ...]
    return AnalysisResponse(
        metrics=metrics,
        visualizations=viz_base64,
        report=enhanced_report,
        evaluation={"score": score, "text": evaluation},
        hyperlinks=all_hyperlinks
    )
