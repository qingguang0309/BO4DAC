# Test script to verify the 4-step workflow
import requests
import json

# Base URL for the application
BASE_URL = "http://127.0.0.1:5001"

def test_api_endpoints():
    print("Testing API endpoints...")
    
    # Test the home page
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Home page: {response.status_code}")
    except Exception as e:
        print(f"Error accessing home page: {e}")
    
    # Test the test API endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/test")
        print(f"Test API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API working: {data.get('success', False)}")
    except Exception as e:
        print(f"Error accessing test API: {e}")
    
    # Test getting status (should work even without initialization)
    try:
        response = requests.get(f"{BASE_URL}/api/get-status")
        print(f"Get status API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Initialized: {data.get('initialized', False)}")
    except Exception as e:
        print(f"Error accessing get-status API: {e}")

if __name__ == "__main__":
    print("Testing the 4-step workflow...")
    test_api_endpoints()
    print("Test completed.")