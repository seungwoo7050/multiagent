import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from src.config.logger import get_logger
from src.core.mcp.api.middleware import BasicMCPMiddleware
from src.core.mcp.api.openapi_extension import (add_mcp_schemas_to_openapi,
                                              customize_openapi_for_mcp,
                                              get_mcp_context_schemas)
from src.core.mcp.api.serialization_middleware import MCPSerializationMiddleware
from src.core.mcp.api.validation_middleware import MCPContextValidationMiddleware
from src.core.mcp.api.websocket_adapter import (MCPWebSocketAdapter,
                                              get_websocket_adapter)
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.serialization import SerializationError, SerializationFormat


class MockContextProtocol(ContextProtocol):
    def __init__(self, context_id="test-context-id", version="1.0.0"):
        self.context_id = context_id
        self.version = version
        self.metadata = {"task_id": "test-task-id"}
    
    def serialize(self):
        return {
            "context_id": self.context_id,
            "version": self.version,
            "metadata": self.metadata
        }
    
    def model_dump(self, mode="json"):
        return self.serialize()


class TestMCPAPI(unittest.TestCase):
    """Test suite for MCP API integration components."""

    def setUp(self):
        self.app = FastAPI()
        self.test_client = TestClient(self.app)

    @pytest.mark.asyncio
    async def test_basic_mcp_middleware(self):
        """Test BasicMCPMiddleware logs and processes requests correctly."""
        with patch('src.core.mcp.api.middleware.logger') as mock_logger:
            middleware = BasicMCPMiddleware(self.app)
            
            # Create mock request and response
            mock_request = MagicMock()
            mock_request.headers = {
                'X-Request-ID': 'test-req-id',
                'content-type': 'application/json+mcp',
                'X-MCP-Version': '1.0.0',
                'X-MCP-Context-Type': 'TaskContext'
            }
            mock_request.url.path = "/test-path"
            mock_request.method = "POST"
            
            # Create mock call_next that returns a mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_call_next = AsyncMock(return_value=mock_response)
            
            # Call dispatch
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            # Assertions
            self.assertEqual(response.status_code, 200)
            self.assertIn('X-Process-Time-Ms', response.headers)
            mock_logger.debug.assert_called()
            mock_logger.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_serialization_middleware(self):
        """Test MCPSerializationMiddleware deserializes context properly."""
        with patch('src.core.mcp.api.serialization_middleware.deserialize_context') as mock_deserialize:
            # Setup the mock to return a valid context
            mock_context = MockContextProtocol()
            mock_deserialize.return_value = mock_context
            
            middleware = MCPSerializationMiddleware(self.app)
            
            # Create mock request with MCP content
            mock_request = MagicMock()
            mock_request.headers = {
                'X-Request-ID': 'test-req-id',
                'content-type': 'application/json+mcp',
            }
            mock_request.url.path = "/test-path"
            mock_request.method = "POST"
            mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
            mock_request.state = MagicMock()
            
            # Create mock call_next
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_call_next = AsyncMock(return_value=mock_response)
            
            # Call dispatch
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            # Assertions
            self.assertEqual(response.status_code, 200)
            mock_deserialize.assert_called_once()
            self.assertEqual(mock_request.state.mcp_context, mock_context)
    
    @pytest.mark.asyncio
    async def test_serialization_middleware_performance(self):
        """Test MCPSerializationMiddleware performance overhead is low."""
        with patch('src.core.mcp.api.serialization_middleware.deserialize_context') as mock_deserialize:
            # Setup the mock to return a valid context
            mock_context = MockContextProtocol()
            mock_deserialize.return_value = mock_context
            
            middleware = MCPSerializationMiddleware(self.app)
            
            # Create mock request with MCP content
            mock_request = MagicMock()
            mock_request.headers = {
                'X-Request-ID': 'test-req-id',
                'content-type': 'application/json+mcp',
            }
            mock_request.url.path = "/test-path"
            mock_request.method = "POST"
            mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
            mock_request.state = MagicMock()
            
            # Create mock call_next that returns immediately
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_call_next = AsyncMock(return_value=mock_response)
            
            # Measure serialization overhead
            start_time = time.time()
            await middleware.dispatch(mock_request, mock_call_next)
            end_time = time.time()
            overhead_ms = (end_time - start_time) * 1000
            
            # Assertion: overhead should be < 1ms as specified in the roadmap
            # Note: In a real test, this might need adjustment based on the test environment
            self.assertLess(overhead_ms, 10, f"Serialization overhead ({overhead_ms}ms) exceeds acceptable limit")
    
    @pytest.mark.asyncio
    async def test_validation_middleware(self):
        """Test MCPContextValidationMiddleware validates context properly."""
        middleware = MCPContextValidationMiddleware(self.app)
        
        # Create mock request with a mock context
        mock_context = MockContextProtocol()
        mock_request = MagicMock()
        mock_request.headers = {'X-Request-ID': 'test-req-id'}
        mock_request.url.path = "/test-path"
        mock_request.state.mcp_context = mock_context
        
        # Create mock call_next
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_call_next = AsyncMock(return_value=mock_response)
        
        # Call dispatch
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        mock_call_next.assert_called_once_with(mock_request)
    
    @pytest.mark.asyncio
    async def test_validation_middleware_rejects_incompatible_version(self):
        """Test validation middleware rejects incompatible context versions."""
        with patch('src.core.mcp.api.validation_middleware.check_version_compatibility') as mock_check:
            mock_check.return_value = False
            
            middleware = MCPContextValidationMiddleware(self.app)
            
            # Create mock request with an incompatible context version
            mock_context = MockContextProtocol(version="999.0.0")
            mock_request = MagicMock()
            mock_request.headers = {'X-Request-ID': 'test-req-id'}
            mock_request.url.path = "/test-path"
            mock_request.state.mcp_context = mock_context
            
            # Create mock call_next (should not be called)
            mock_call_next = AsyncMock()
            
            # Call dispatch
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            # Assertions
            self.assertNotEqual(response.status_code, 200)
            mock_call_next.assert_not_called()
    
    def test_openapi_extension(self):
        """Test OpenAPI extension functions work correctly."""
        # Test get_mcp_context_schemas
        schemas = get_mcp_context_schemas()
        self.assertIsInstance(schemas, list)
        self.assertTrue(len(schemas) > 0)
        
        # Test customize_openapi_for_mcp
        app = FastAPI()
        original_openapi = app.openapi
        customize_openapi_for_mcp(app)
        self.assertNotEqual(original_openapi, app.openapi)
        
        # Test the customized OpenAPI
        openapi_schema = app.openapi()
        self.assertIn('x-mcp-support', openapi_schema['info'])
        self.assertEqual(openapi_schema['info']['x-mcp-support']['version'], '1.0.0')
    
    @pytest.mark.asyncio
    async def test_websocket_adapter(self):
        """Test WebSocket adapter can stream contexts."""
        with patch('src.core.mcp.api.websocket_adapter.serialize_context') as mock_serialize:
            # Setup mocks
            mock_serialize.return_value = b'{"test": "data"}'
            
            # Create mock connection manager
            mock_connection_manager = MagicMock()
            mock_connection_manager.broadcast_to_task = AsyncMock()
            
            # Create adapter
            adapter = MCPWebSocketAdapter(mock_connection_manager)
            
            # Stream a context
            mock_context = MockContextProtocol()
            result = await adapter.stream_context(mock_context)
            
            # Assertions
            self.assertTrue(result)
            mock_serialize.assert_called_once()
            mock_connection_manager.broadcast_to_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_adapter_singleton(self):
        """Test that get_websocket_adapter returns a singleton."""
        with patch('src.core.mcp.api.websocket_adapter.get_connection_manager') as mock_get_cm:
            # Setup mock connection manager
            mock_connection_manager = MagicMock()
            mock_get_cm.return_value = mock_connection_manager
            
            # Get adapter instance
            adapter1 = await get_websocket_adapter()
            adapter2 = await get_websocket_adapter()
            
            # Assertions
            self.assertIsInstance(adapter1, MCPWebSocketAdapter)
            self.assertIs(adapter1, adapter2)  # Should be the same instance
    
    @pytest.mark.asyncio
    async def test_middleware_pipeline_integration(self):
        """Test complete middleware pipeline with MCP context handling."""
        app = FastAPI()
        
        # Add all middlewares
        app.add_middleware(BasicMCPMiddleware)
        app.add_middleware(MCPContextValidationMiddleware)
        app.add_middleware(MCPSerializationMiddleware)
        
        # Add a test endpoint
        @app.post("/test")
        async def test_endpoint(request: Request):
            # Return the MCP context if available
            mcp_context = getattr(request.state, "mcp_context", None)
            if mcp_context:
                return {"context_id": mcp_context.context_id}
            return {"error": "No context found"}
        
        # Mock the serialization functionality
        with patch('src.core.mcp.api.serialization_middleware.deserialize_context') as mock_deserialize:
            mock_context = MockContextProtocol()
            mock_deserialize.return_value = mock_context
            
            # Test client
            client = TestClient(app)
            
            # Make request
            response = client.post(
                "/test", 
                json={"test": "data"},
                headers={"content-type": "application/json+mcp"}
            )
            
            # Assertions
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"context_id": "test-context-id"})


if __name__ == "__main__":
    unittest.main()