from typing import Any, Dict

from packaging import version

from src.config.logger import get_logger

logger = get_logger(__name__)
SUPPORTED_VERSIONS = ['1.0.0']

def check_version_compatibility(context_version_str: str) -> bool:
    try:
        parsed_context_version = version.parse(context_version_str)
        latest_supported_version_str = get_latest_supported_version()
        latest_supported_version = version.parse(latest_supported_version_str)
        is_compatible = parsed_context_version.major == latest_supported_version.major and parsed_context_version >= latest_supported_version
        logger.debug(f"Version compatibility check: Context='{context_version_str}', LatestSupported='{latest_supported_version_str}', Compatible={is_compatible}")
        return is_compatible
    except version.InvalidVersion:
        logger.warning(f'Invalid context version format encountered: {context_version_str}')
        return False
    except Exception as e:
        logger.error(f'Error during version compatibility check for {context_version_str}: {e}')
        return False

def upgrade_context(context_data: Dict[str, Any], target_version_str: str) -> Dict[str, Any]:
    current_version_str = context_data.get('version', '0.0.0')
    current_version = version.parse(current_version_str)
    target_version = version.parse(target_version_str)
    
    if current_version == target_version:
        logger.debug(f'Context is already at target version {target_version_str}. No upgrade needed.')
        return context_data
    
    if current_version > target_version:
        logger.warning(f'Attempting to downgrade context from {current_version} to {target_version}. Not supported.')
        return context_data
    
    # Copy the context data to avoid modifying the original
    upgraded_data = dict(context_data)
    
    # Implementation for version upgrade paths
    if current_version_str == '0.0.0' and target_version_str in SUPPORTED_VERSIONS:
        # Upgrade from unversioned context to supported version
        upgraded_data['version'] = target_version_str
        logger.info(f'Upgraded unversioned context to {target_version_str}')
        return upgraded_data
    
    # Add other upgrade paths based on specific version transitions
    
    logger.error(f'Unsupported context upgrade path from {current_version_str} to {target_version_str}')
    raise ValueError(f'Cannot upgrade context from version {current_version_str} to {target_version_str}')

def get_latest_supported_version() -> str:
    return max(SUPPORTED_VERSIONS, key=version.parse)