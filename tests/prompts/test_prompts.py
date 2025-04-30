import pytest
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the modules to be tested
import src.prompts.templates as templates
from src.prompts.templates import PromptTemplate, PromptTemplateLoader


class TestPromptTemplate:
    def test_init_with_variables(self):
        template = PromptTemplate(
            "test_template", 
            "Hello {name}, welcome to {service}!",
            ["name", "service"]
        )
        assert template.template_id == "test_template"
        assert template.template_string == "Hello {name}, welcome to {service}!"
        assert template.input_variables == {"name", "service"}

    def test_init_without_variables(self):
        template = PromptTemplate(
            "test_template", 
            "Hello {name}, welcome to {service}!"
        )
        assert template.input_variables == {"name", "service"}

    def test_extract_variables(self):
        template = PromptTemplate("test", "Test {simple} {nested.property} {indexed[0]}")
        assert template.input_variables == {"simple", "nested", "indexed"}

    def test_validate_template_valid(self):
        template = PromptTemplate("valid", "Valid {template} with {variables}")
        assert template._validate_template() is True

    def test_validate_template_unbalanced(self):
        template = PromptTemplate("unbalanced", "Unbalanced {template with missing brace")
        assert template._validate_template() is False

    def test_validate_template_escaped_braces(self):
        template = PromptTemplate("escaped", "Template with {{escaped}} braces and {real_var}")
        assert template._validate_template() is True
        assert template.input_variables == {"real_var"}

    def test_render_success(self):
        template = PromptTemplate("success", "Hello {name}!")
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_render_missing_variable(self):
        template = PromptTemplate("missing", "Hello {name}!")
        with pytest.raises(KeyError) as excinfo:
            template.render()
        assert "Missing required variables" in str(excinfo.value)

    def test_render_complex_expressions(self):
        template = PromptTemplate(
            "complex", 
            "Name: {person.name}, Age: {person.age}, First hobby: {person.hobbies[0]}"
        )
        result = template.render(
            person={
                "name": "Alice", 
                "age": 30, 
                "hobbies": ["reading", "hiking"]
            }
        )
        assert result == "Name: Alice, Age: 30, First hobby: reading"


class TestPromptTemplateLoader:
    @pytest.fixture
    def template_loader(self):
        return PromptTemplateLoader("test_templates/")

    @pytest.mark.asyncio
    async def test_load_hardcoded_template(self, template_loader):
        template = await template_loader.load_template("planner_prompt")
        assert template is not None
        assert template.template_id == "planner_prompt"
        assert "**Goal:** {goal}" in template.template_string

    @pytest.mark.asyncio
    async def test_cache_behavior(self, template_loader):
        # Load a template to cache it
        template1 = await template_loader.load_template("planner_prompt")
        
        # Should retrieve from cache on second call
        with patch.object(template_loader, '_get_hardcoded_template') as mock_get:
            template2 = await template_loader.load_template("planner_prompt")
            mock_get.assert_not_called()
        
        # Both references should be to the same object
        assert template1 is template2

    @pytest.mark.asyncio
    async def test_load_from_json_file(self, template_loader):
        json_content = {
            "test_template": {
                "template": "This is a {test} template",
                "input_variables": ["test"]
            }
        }
        
        # Mock the file reading
        with patch.object(template_loader, '_read_json_file', return_value=json_content):
            with patch('pathlib.Path.exists', return_value=True):
                template = await template_loader._load_from_json_file(
                    Path("test_templates/templates.json"), 
                    "test_template",
                    key_in_file=True
                )
                
                assert template is not None
                assert template.template_id == "test_template"
                assert template.template_string == "This is a {test} template"
                assert template.input_variables == {"test"}

    @pytest.mark.asyncio
    async def test_load_from_txt_file(self, template_loader):
        txt_content = "This is a {test} template"
        
        # Mock the file reading
        with patch.object(template_loader, '_read_text_file', return_value=txt_content):
            template = await template_loader._load_from_txt_file(
                Path("test_templates/test_template.txt"), 
                "test_template"
            )
            
            assert template is not None
            assert template.template_id == "test_template"
            assert template.template_string == "This is a {test} template"
            assert template.input_variables == {"test"}

    @pytest.mark.asyncio
    async def test_load_template_not_found(self, template_loader):
        with patch.object(template_loader, '_load_from_source', return_value=None):
            with patch.object(template_loader, '_get_hardcoded_template', return_value=None):
                template = await template_loader.load_template("nonexistent_template")
                assert template is None


class TestGlobalLoader:
    @pytest.mark.asyncio
    async def test_get_prompt_loader(self):
        # Reset the global loader
        templates._default_loader = None
        
        # Get the loader with a custom source
        loader1 = await templates.get_prompt_loader("custom_source/")
        assert loader1.template_source == "custom_source/"
        
        # Get the loader again, should be the same instance
        loader2 = await templates.get_prompt_loader("different_source/")
        assert loader2 is loader1
        assert loader2.template_source == "custom_source/"  # Source shouldn't change

    @pytest.mark.asyncio
    async def test_reload_templates(self):
        # Reset the global loader
        templates._default_loader = None
        
        # Create a loader and add something to the cache
        loader = await templates.get_prompt_loader("test_source/")
        loader._cache["test"] = PromptTemplate("test", "Test template")
        
        # Reload templates should clear the cache
        await templates.reload_templates()
        assert "test" not in loader._cache


# Integration test
class TestTemplateIntegration:
    @pytest.fixture
    def setup_template_files(self, tmp_path):
        # Create a temporary directory structure
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        
        # Create a JSON file with multiple templates
        templates_json = templates_dir / "templates.json"
        templates_json.write_text(json.dumps({
            "greeting": {
                "template": "Hello {name}, welcome to {service}!",
                "input_variables": ["name", "service"]
            },
            "farewell": "Goodbye {name}, thank you for using {service}!"
        }))
        
        # Create a single template file
        single_template = templates_dir / "single_template.json"
        single_template.write_text(json.dumps({
            "template": "This is a {standalone} template",
            "input_variables": ["standalone"]
        }))
        
        # Create a text template
        text_template = templates_dir / "text_template.txt"
        text_template.write_text("This is a plain {text} template")
        
        return str(templates_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end(self, setup_template_files):
        # Reset the global loader
        templates._default_loader = None
        
        # Get the loader with our temp directory
        loader = await templates.get_prompt_loader(setup_template_files)
        
        # Test loading and rendering each template type
        greeting = await loader.load_template("greeting")
        assert greeting is not None
        rendered = greeting.render(name="Alice", service="TestService")
        assert rendered == "Hello Alice, welcome to TestService!"
        
        farewell = await loader.load_template("farewell")
        assert farewell is not None
        rendered = farewell.render(name="Alice", service="TestService")
        assert rendered == "Goodbye Alice, thank you for using TestService!"
        
        # Should retrieve from cache on second call
        with patch.object(loader, '_load_from_source'):
            greeting2 = await loader.load_template("greeting")
            assert greeting2 is greeting

if __name__ == "__main__":
    pytest.main()