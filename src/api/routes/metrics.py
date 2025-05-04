# --- 교체할 get_metrics 핸들러 ---------------------------------------
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, REGISTRY
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse


router = APIRouter(tags=["Metrics"])
logger = get_logger(__name__)

from fastapi import Header

@router.get("/metrics", status_code=200)
async def get_metrics(accept: str = Header(default="text/plain")):
    try:
        prom_blob = generate_latest(REGISTRY)
        
        # If client wants JSON format
        if "application/json" in accept.lower():
            try:
                # Safely parse metrics to a flat dictionary
                flat = {}
                for line in prom_blob.decode().splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            # Handle metric name with labels
                            key_parts = parts[0].split("{")
                            metric_name = key_parts[0]
                            
                            # Try to parse the value as float, but don't fail if it's not
                            try:
                                value = float(parts[-1])
                            except (ValueError, TypeError):
                                value = parts[-1]  # Keep as string if can't convert
                                
                            flat[metric_name] = value
                    except Exception as parse_err:
                        logger.warning(f"Could not parse metric line: {line}, error: {parse_err}")
                        continue
                
                # Try to get grouped metrics if metrics manager is available
                try:
                    mm = get_metrics_manager()
                    if mm:
                        grouped = {
                            "task_metrics": mm.TASK_METRICS,
                            "agent_metrics": mm.AGENT_METRICS,
                            "tool_metrics": mm.TOOL_METRICS,
                            "raw": flat,
                        }
                        return JSONResponse(grouped)
                except Exception as mm_err:
                    logger.warning(f"Error getting metrics manager: {mm_err}")
                    # Fall back to flat metrics
                    return JSONResponse({"metrics": flat})
                    
                # Fall back to flat metrics if mm wasn't available
                return JSONResponse({"metrics": flat})
            except Exception as json_err:
                logger.error(f"Error creating JSON metrics response: {json_err}")
                # Fall back to text format if JSON conversion fails
                return PlainTextResponse(content=prom_blob, media_type=CONTENT_TYPE_LATEST)
                
        # Default case: return plain text Prometheus format
        return PlainTextResponse(content=prom_blob, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}", exc_info=True)
        # Instead of 500, return empty metrics with 200
        return JSONResponse(
            status_code=200,
            content={"metrics": {}, "error": "Failed to generate metrics"}
        )