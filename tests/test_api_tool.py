"""Test script for API Tool - demonstrates mock API calls."""
import sys
from pathlib import Path

# Ensure project root import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.api_tool import create_api_tool


def test_order_status():
    """Test order status query."""
    print("\n" + "="*80)
    print("TEST 1: Order Status Query")
    print("="*80)
    
    api_tool = create_api_tool(mock_delay=0.1)
    
    # Method 1: JSON input
    query = '{"query_type": "order_status", "parameters": {"order_id": "ORD12345"}}'
    result = api_tool._run(query)
    print(f"\nQuery: {query}")
    print(f"\nResult:\n{result}")


def test_inventory():
    """Test inventory check."""
    print("\n" + "="*80)
    print("TEST 2: Inventory Check")
    print("="*80)
    
    api_tool = create_api_tool(mock_delay=0.1)
    
    # Method 2: Plain text (auto-detection)
    query = "check inventory for product PROD001"
    result = api_tool._run(query)
    print(f"\nQuery: {query}")
    print(f"\nResult:\n{result}")


def test_product_info():
    """Test product information query."""
    print("\n" + "="*80)
    print("TEST 3: Product Information")
    print("="*80)
    
    api_tool = create_api_tool(mock_delay=0.1)
    
    query = '{"query_type": "product_info", "parameters": {"product_id": "PROD001"}}'
    result = api_tool._run(query)
    print(f"\nQuery: {query}")
    print(f"\nResult:\n{result}")


def test_service_status():
    """Test service status check."""
    print("\n" + "="*80)
    print("TEST 4: Service Status")
    print("="*80)
    
    api_tool = create_api_tool(mock_delay=0.1)
    
    query = "what is the service status"
    result = api_tool._run(query)
    print(f"\nQuery: {query}")
    print(f"\nResult:\n{result}")


def test_invalid_query():
    """Test invalid query handling."""
    print("\n" + "="*80)
    print("TEST 5: Invalid Query")
    print("="*80)
    
    api_tool = create_api_tool(mock_delay=0.1)
    
    query = "this is a random query"
    result = api_tool._run(query)
    print(f"\nQuery: {query}")
    print(f"\nResult:\n{result}")


if __name__ == "__main__":
    print("\n🧪 Testing API Tool - Mock API Calls\n")
    
    test_order_status()
    test_inventory()
    test_product_info()
    test_service_status()
    test_invalid_query()
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80 + "\n")

