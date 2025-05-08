# tests/test_services.py

import importlib
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# --- 테스트 대상 임포트 ---
from src.services.tool_manager import ToolManager, get_tool_manager, register_tool
from src.tools.base import BaseTool
from pydantic import BaseModel, Field # MockTool 정의 위해 필요

# --- Fixtures ---

# MockTool 클래스 정의 (이 파일 내에서 사용)
class MockToolInput(BaseModel):
    arg1: str = Field(...)

class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing ToolManager."
    args_schema: type[BaseModel] = MockToolInput
    def _run(self, arg1: str) -> str: return f"Mock run: {arg1}"
    async def _arun(self, arg1: str) -> str: return f"Mock arun: {arg1}"

@pytest.fixture(scope="function")
def tool_manager_instance():
    """새로운 ToolManager 인스턴스를 생성합니다."""
    manager = ToolManager(name=f"test_manager_{os.urandom(4).hex()}")
    return manager

@pytest.fixture
def mock_tools_dir(tmp_path):
    """테스트용 도구 모듈 파일을 임시 디렉토리에 생성합니다."""
    tools_dir = tmp_path / "mock_tools_pkg_services" # 이전과 다른 이름 사용
    tools_dir.mkdir()
    (tools_dir / "__init__.py").touch()

    # Mock Tool A (정상)
    (tools_dir / "tool_a.py").write_text("""
from src.tools.base import BaseTool
# 전역 register_tool 사용 가정
from src.services.tool_manager import register_tool
from pydantic import BaseModel, Field

class ToolAInput(BaseModel):
    x: int

# @register_tool() # 테스트 시에는 데코레이터 주석 처리 또는 패치 필요
class ToolA(BaseTool):
    name = "tool_a"
    description = "Tool A description"
    args_schema = ToolAInput

    def _run(self, x: int) -> str:
        return f"Tool A executed with {x}"
""")
    # Mock Tool B (정상)
    (tools_dir / "tool_b.py").write_text("""
from src.tools.base import BaseTool
from pydantic import BaseModel

# @register_tool()
class ToolB(BaseTool):
    name = "tool_b"
    description = "Tool B description"
    args_schema = BaseModel

    def _run(self) -> str:
        return "Tool B executed"
""")
    # BaseTool 상속 안 함
    (tools_dir / "not_a_tool.py").write_text("class NotATool: pass")

    return tools_dir

# --- ToolManager 테스트 ---

def test_tool_manager_registration(tool_manager_instance):
    """ToolManager에 도구 클래스를 직접 등록하는 기능 테스트"""
    assert tool_manager_instance.name != "global_tools"
    tool_manager_instance.register(MockTool)
    assert tool_manager_instance.has("mock_tool") # has 메서드 사용
    assert tool_manager_instance.get_tool_class("mock_tool") == MockTool

def test_tool_manager_dynamic_loading(tool_manager_instance, mock_tools_dir):
    """ToolManager가 디렉토리에서 도구를 동적으로 로드하는 기능 테스트"""
    parent_dir = str(mock_tools_dir.parent)
    original_sys_path = sys.path[:]
    path_added_to_sys = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        path_added_to_sys = True

    # load_tools_from_directory 호출 시 절대 경로 문자열 전달
    imported_count = tool_manager_instance.load_tools_from_directory(str(mock_tools_dir))

    # sys.path 복원
    if path_added_to_sys:
        try:
            sys.path.remove(parent_dir)
        except ValueError:
            pass # 이미 제거된 경우 무시

    # 검증: 모듈 임포트 성공 횟수 확인
    assert imported_count >= 2, f"Expected at least 2 modules imported, but got {imported_count}"

    # 검증: 로드된 도구가 (기본적으로) 전역 ToolManager에 등록되었는지 확인
    #       또는 데코레이터가 테스트용 매니저를 사용하도록 패치 필요
    # 여기서는 전역 매니저 확인 (get_tool_manager() 사용)
    global_manager = get_tool_manager('global_tools')
    assert global_manager.has("tool_a"), "ToolA should be registered in global manager"
    assert global_manager.has("tool_b"), "ToolB should be registered in global manager"
    assert not global_manager.has("not_a_tool")


def test_tool_manager_get_tool(tool_manager_instance):
    """ToolManager에서 도구 인스턴스를 가져오는 기능 테스트 (캐싱 포함)"""
    tool_manager_instance.register(MockTool) # 파일 내에 정의된 MockTool 사용

    instance1 = tool_manager_instance.get_tool("mock_tool")
    assert isinstance(instance1, MockTool)
    assert "mock_tool" in tool_manager_instance._instance_cache

    instance2 = tool_manager_instance.get_tool("mock_tool")
    assert instance1 is instance2

def test_tool_manager_list_tools_and_summary(tool_manager_instance):
    """ToolManager의 도구 목록 및 LLM용 요약 정보 생성 테스트"""
    from src.tools.calculator import CalculatorTool # 실제 도구 임포트
    tool_manager_instance.register(MockTool) # 파일 내에 정의된 MockTool 사용
    tool_manager_instance.register(CalculatorTool)

    tool_list = tool_manager_instance.list_tools()
    assert len(tool_list) == 2
    mock_tool_info = next((t for t in tool_list if t['name'] == 'mock_tool'), None)
    calc_tool_info = next((t for t in tool_list if t['name'] == 'calculator'), None)

    assert mock_tool_info is not None
    assert mock_tool_info['args_schema_summary'] == {'arg1': 'string (required)'} # MockToolInput 스키마 반영

    assert calc_tool_info is not None
    assert calc_tool_info['args_schema_summary'] == {'expression': 'string (required)'}

    llm_summaries = tool_manager_instance.get_tool_summaries_for_llm()
    assert len(llm_summaries) == 2
    mock_tool_summary = next((s for s in llm_summaries if s['name'] == 'mock_tool'), None)

    assert mock_tool_summary is not None
    assert "(Arguments: arg1: string (required))" in mock_tool_summary['description']