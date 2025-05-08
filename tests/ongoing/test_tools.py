# tests/test_tools.py

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from pydantic import BaseModel, Field

# --- 테스트 대상 임포트 ---
from src.tools.base import BaseTool
from src.tools.calculator import CalculatorTool, CalculatorInput
from src.tools.datetime_tool import DateTimeTool, DateTimeInput, DateTimeOperation
from src.tools.web_search import WebSearchTool, DuckDuckGoInput
from src.config.errors import ToolError

# --- BaseTool 계약 테스트 ---

class MockToolInput(BaseModel):
    arg1: str = Field(..., description="Argument 1")
    arg2: int = Field(default=5, description="Argument 2")

class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing BaseTool contract."
    args_schema: type[BaseModel] = MockToolInput

    def _run(self, arg1: str, arg2: int = 5) -> str:
        """Mock sync execution."""
        return f"Mock Result: {arg1}, {arg2}"

    async def _arun(self, arg1: str, arg2: int = 5) -> str:
        """Mock async execution."""
        await asyncio.sleep(0.01) # Simulate async work
        return f"Mock Async Result: {arg1}, {arg2}"

def test_base_tool_contract():
    """BaseTool 인터페이스 계약 (속성 존재 여부) 테스트"""
    mock_tool = MockTool()
    assert mock_tool.name == "mock_tool"
    assert mock_tool.description == "A mock tool for testing BaseTool contract."
    assert mock_tool.args_schema == MockToolInput
    assert hasattr(mock_tool, "_run") and callable(mock_tool._run)
    assert hasattr(mock_tool, "_arun") and callable(mock_tool._arun)

@pytest.mark.asyncio
async def test_base_tool_execution():
    """BaseTool 기본 실행 (_run, _arun) 테스트"""
    mock_tool = MockTool()

    # Sync 실행 테스트
    sync_result = mock_tool._run(arg1="hello")
    assert sync_result == "Mock Result: hello, 5"
    sync_result_with_arg = mock_tool._run(arg1="world", arg2=10)
    assert sync_result_with_arg == "Mock Result: world, 10"

    # Async 실행 테스트
    async_result = await mock_tool._arun(arg1="async_hello")
    assert async_result == "Mock Async Result: async_hello, 5"
    async_result_with_arg = await mock_tool._arun(arg1="async_world", arg2=15)
    assert async_result_with_arg == "Mock Async Result: async_world, 15"

# --- 개별 도구 기능 테스트 ---

# CalculatorTool 테스트
def test_calculator_tool_simple_evaluation():
    """CalculatorTool 기본 연산 성공 테스트"""
    calculator = CalculatorTool()
    expression = "2 + 3 * 4"
    result = calculator._run(expression=expression)
    # 안전한 eval 결과는 문자열이어야 함
    assert isinstance(result, str)
    # 부동소수점 비교를 위해 float으로 변환
    assert float(result) == 14.0

@pytest.mark.asyncio
async def test_calculator_tool_async_wrapper():
    """CalculatorTool async 래퍼 성공 테스트"""
    calculator = CalculatorTool()
    expression = "sqrt(16)"
    result = await calculator._arun(expression=expression)
    assert isinstance(result, str)
    assert float(result) == 4.0

# DateTimeTool 테스트
@pytest.mark.asyncio
async def test_datetime_tool_current_time():
    """DateTimeTool 'current' 연산 성공 테스트"""
    datetime_tool = DateTimeTool()
    # 비동기 실행 테스트
    result_str = await datetime_tool._arun(operation=DateTimeOperation.CURRENT, timezone='UTC')
    import json
    result = json.loads(result_str)
    assert result["operation"] == "current"
    assert result["timezone"] == "UTC"
    assert "iso_format" in result
    assert isinstance(result["timestamp"], float)

# WebSearchTool 테스트
@pytest.mark.asyncio
async def test_web_search_tool_success(mocker):
    """WebSearchTool 성공적인 실행 (DuckDuckGoSearchRun 모킹) 테스트"""
    # DuckDuckGoSearchRun 인스턴스와 해당 메서드를 모킹
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.run = MagicMock(return_value="Mock search result for 'test query'")
    mock_ddg_instance.arun = AsyncMock(return_value="Mock async search result for 'async query'")

    # WebSearchTool이 DuckDuckGoSearchRun을 초기화할 때 모킹된 인스턴스를 반환하도록 패치
    mocker.patch('src.tools.web_search.DuckDuckGoSearchRun', return_value=mock_ddg_instance)

    web_search_tool = WebSearchTool()

    # Sync 실행 테스트
    sync_result = web_search_tool._run(query="test query")
    assert sync_result == "Mock search result for 'test query'"
    mock_ddg_instance.run.assert_called_once_with(tool_input="test query")

    # Async 실행 테스트
    async_result = await web_search_tool._arun(query="async query")
    assert async_result == "Mock async search result for 'async query'"
    mock_ddg_instance.arun.assert_awaited_once_with(tool_input="async query")