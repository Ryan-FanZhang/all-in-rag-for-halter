"""API Tool: Mock external API calls for live data retrieval."""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta
from typing import Optional

from langchain.tools import BaseTool
from pydantic import Field


class APITool(BaseTool):
    """Tool for calling external APIs to retrieve live data.
    
    This is a mock implementation for testing. In production, replace with
    actual API calls to your backend services.
    """

    name: str = "api_call"
    description: str = (
        "Use this tool to retrieve live data from external systems. "
        "Supports queries like: order status, inventory check, product info, service status. "
        "Input should be a JSON string with 'query_type' and 'parameters'."
    )
    
    # Configuration
    mock_delay: float = Field(default=0.5, description="Simulated API delay in seconds")

    def _run(self, query_info: str) -> str:
        """Execute mock API call and return live data."""
        try:
            # Simulate API delay
            time.sleep(self.mock_delay)
            
            # Parse input
            if query_info.strip().startswith('{'):
                try:
                    query = json.loads(query_info)
                    query_type = query.get("query_type", "")
                    params = query.get("parameters", {})
                except json.JSONDecodeError:
                    query_type = "unknown"
                    params = {}
            else:
                # Try to infer query type from plain text
                query_text = query_info.lower()
                if "order" in query_text or "订单" in query_text:
                    query_type = "order_status"
                    params = {"order_id": "ORD12345"}
                elif "inventory" in query_text or "库存" in query_text or "stock" in query_text:
                    query_type = "inventory"
                    params = {"product_id": "PROD001"}
                elif "product" in query_text or "产品" in query_text:
                    query_type = "product_info"
                    params = {"product_id": "PROD001"}
                elif "service" in query_text or "status" in query_text or "服务" in query_text:
                    query_type = "service_status"
                    params = {}
                else:
                    query_type = "unknown"
                    params = {}
            
            # Route to appropriate mock handler
            if query_type == "order_status":
                result = self._mock_order_status(params)
            elif query_type == "inventory":
                result = self._mock_inventory(params)
            elif query_type == "product_info":
                result = self._mock_product_info(params)
            elif query_type == "service_status":
                result = self._mock_service_status(params)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown query type: {query_type}",
                    "supported_types": ["order_status", "inventory", "product_info", "service_status"]
                }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"API call failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)

    def _mock_order_status(self, params: dict) -> dict:
        """Mock order status API."""
        order_id = params.get("order_id", "ORD" + str(random.randint(10000, 99999)))
        
        statuses = ["pending", "processing", "shipped", "delivered"]
        status = random.choice(statuses)
        
        # Generate realistic timeline
        created_at = datetime.now() - timedelta(days=random.randint(1, 7))
        
        result = {
            "status": "success",
            "query_type": "order_status",
            "data": {
                "order_id": order_id,
                "status": status,
                "created_at": created_at.isoformat(),
                "items": [
                    {
                        "product_id": "PROD001",
                        "product_name": "Coffee Machine ECAM23.420",
                        "quantity": 1,
                        "price": 599.99
                    }
                ],
                "total_amount": 599.99,
                "shipping_address": {
                    "city": "New York",
                    "country": "USA"
                },
                "tracking_number": f"TRK{random.randint(100000000, 999999999)}" if status in ["shipped", "delivered"] else None,
                "estimated_delivery": (datetime.now() + timedelta(days=random.randint(1, 5))).date().isoformat() if status != "delivered" else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _mock_inventory(self, params: dict) -> dict:
        """Mock inventory check API."""
        product_id = params.get("product_id", "PROD" + str(random.randint(100, 999)))
        
        result = {
            "status": "success",
            "query_type": "inventory",
            "data": {
                "product_id": product_id,
                "product_name": "Coffee Machine ECAM23.420",
                "in_stock": random.choice([True, False]),
                "quantity": random.randint(0, 100),
                "warehouse_locations": [
                    {"location": "Warehouse A", "quantity": random.randint(0, 50)},
                    {"location": "Warehouse B", "quantity": random.randint(0, 50)}
                ],
                "restock_date": (datetime.now() + timedelta(days=random.randint(7, 30))).date().isoformat() if random.random() < 0.3 else None,
                "last_updated": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _mock_product_info(self, params: dict) -> dict:
        """Mock product information API."""
        product_id = params.get("product_id", "PROD001")
        
        result = {
            "status": "success",
            "query_type": "product_info",
            "data": {
                "product_id": product_id,
                "name": "De'Longhi ECAM23.420 Coffee Machine",
                "category": "Coffee Machines",
                "price": 599.99,
                "currency": "USD",
                "description": "Bean to cup espresso and cappuccino machine with adjustable grinder",
                "specifications": {
                    "voltage": "220-240V",
                    "power": "1450W",
                    "pressure": "15 bar",
                    "water_tank": "1.4L",
                    "bean_container": "200g"
                },
                "availability": "In Stock",
                "rating": 4.5,
                "reviews_count": 1234,
                "manufacturer": "De'Longhi",
                "warranty": "2 years"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _mock_service_status(self, params: dict) -> dict:
        """Mock service status API."""
        # Simulate checking various service health
        services = {
            "payment_gateway": random.choice(["operational", "degraded", "down"]),
            "inventory_system": random.choice(["operational", "degraded"]),
            "shipping_api": random.choice(["operational", "operational", "degraded"]),
            "notification_service": "operational"
        }
        
        overall_status = "operational"
        if "down" in services.values():
            overall_status = "down"
        elif "degraded" in services.values():
            overall_status = "degraded"
        
        result = {
            "status": "success",
            "query_type": "service_status",
            "data": {
                "overall_status": overall_status,
                "services": services,
                "uptime_percentage": round(random.uniform(98.5, 99.9), 2),
                "last_incident": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() if random.random() < 0.3 else None,
                "maintenance_scheduled": (datetime.now() + timedelta(days=random.randint(7, 30))).isoformat() if random.random() < 0.2 else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    async def _arun(self, query_info: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query_info)


def create_api_tool(mock_delay: float = 0.5) -> APITool:
    """Factory function to create a configured API tool."""
    return APITool(mock_delay=mock_delay)

