# src/api/routes/ with test_agent_config.json
"""
Test script for Agent API endpoints
Usage: python test_agent_api.py
"""
import requests
import json
import sys
import time

# Config
API_BASE_URL = "http://127.0.0.1:8000/api/v1"  # Adjust port if needed
AGENTS_ENDPOINT = f"{API_BASE_URL}/agents"

def test_list_agents():
    """Test GET /agents endpoint"""
    print("\n=== Testing GET /agents ===")
    response = requests.get(AGENTS_ENDPOINT)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        agents = response.json()
        print(f"Found {len(agents)} registered agents:")
        for agent in agents:
            print(f"  - {agent['name']} (Type: {agent['agent_type']})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_register_agent(config_file):
    """Test POST /agents endpoint with config from file"""
    print(f"\n=== Testing POST /agents with {config_file} ===")
    
    # Load config from file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration for agent: {config.get('name', 'unknown')}")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return False
    
    # Send POST request
    try:
        response = requests.post(AGENTS_ENDPOINT, json=config)
        
        print(f"Status: {response.status_code}")
        if response.status_code in (200, 201):
            result = response.json()
            print(f"Successfully registered agent: {result['name']} (Type: {result['agent_type']})")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_get_agent_config(agent_name):
    """Test GET /agents/{agent_name} endpoint"""
    print(f"\n=== Testing GET /agents/{agent_name} ===")
    
    response = requests.get(f"{AGENTS_ENDPOINT}/{agent_name}")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        config = response.json()
        print(f"Retrieved configuration for agent: {config['name']}")
        print(f"  Type: {config['agent_type']}")
        print(f"  Model: {config['model']}")
        print(f"  Description: {config.get('description', 'N/A')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def run_tests():
    """Run all tests"""
    print("Starting API tests...")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Check if API is available
    try:
        requests.get(API_BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {API_BASE_URL}")
        print("Make sure the server is running with: uvicorn src.api.app:app --reload")
        return False
    
    # Run tests
    test_list_agents()
    
    # Register a new agent
    if test_register_agent("test_agent_config.json"):
        # Get the registered agent config
        time.sleep(1)  # Small delay to ensure consistency
        test_get_agent_config("test_calculator_agent")
    
    print("\nTests completed.")
    return True

if __name__ == "__main__":
    run_tests()