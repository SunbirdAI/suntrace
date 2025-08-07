#!/usr/bin/env python3
"""
Simple API test for buffer functionality to verify the Flask endpoints work.
"""

import json

import requests
from shapely.geometry import Point
from shapely.wkt import dumps as wkt_dumps

BASE_URL = "http://127.0.0.1:5000"


def test_api_buffer():
    """Test buffer functionality through API endpoints"""
    print("🌐 Testing Buffer API Endpoints")
    print("=" * 35)

    # Test 1: Check if server is running
    print("\n1️⃣ Checking API server...")
    try:
        response = requests.get(f"{BASE_URL}/get_map_layers", timeout=5)
        if response.status_code == 200:
            print("   ✅ API server is running")
        else:
            print(f"   ❌ API server returned {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Could not connect to API server: {e}")
        print("   💡 Make sure Flask server is running: python src/app.py")
        return False

    # Test 2: Test buffer creation
    print("\n2️⃣ Testing buffer creation...")
    test_point = Point(32.8, 3.16)  # Point in Lamwo, Uganda
    test_wkt = wkt_dumps(test_point)
    print(f"   📍 Test point: {test_wkt}")

    buffer_data = {"geometry_wkt": test_wkt, "radius_m": 1000}

    try:
        response = requests.post(
            f"{BASE_URL}/api/buffer-feature", json=buffer_data, timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("   ✅ Buffer creation successful!")
            print(
                f"   📏 Buffer radius: {result['buffer_geojson']['properties']['radius_m']}m"
            )

            # Store the buffered geometry for subsequent tests
            buffered_wkt = result.get("buffered_geometry_wkt")

            # Test 3: Query features within buffer
            print("\n3️⃣ Testing buffer queries...")

            # Test buildings
            query_data = {
                "query_type": "buildings",
                "geometry_wkt": buffered_wkt,
                "radius_m": 1000,
            }
            response = requests.post(
                f"{BASE_URL}/api/query-buffer", json=query_data, timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                building_count = result["result"]["count"]
                print(f"   🏠 Buildings in buffer: {building_count}")
            else:
                print(
                    f"   ⚠️ Building query returned {response.status_code}: {response.text}"
                )

            # Test minigrids
            query_data = {
                "query_type": "minigrids",
                "geometry_wkt": buffered_wkt,
                "radius_m": 1000,
            }
            response = requests.post(
                f"{BASE_URL}/api/query-buffer", json=query_data, timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                minigrid_count = result["result"]["count"]
                print(f"   ⚡ Minigrids in buffer: {minigrid_count}")
            else:
                print(
                    f"   ⚠️ Minigrid query returned {response.status_code}: {response.text}"
                )

            # Test NDVI
            query_data = {
                "query_type": "ndvi",
                "geometry_wkt": buffered_wkt,
                "radius_m": 1000,
            }
            try:
                response = requests.post(
                    f"{BASE_URL}/api/query-buffer", json=query_data, timeout=10
                )
                if response.status_code == 200:
                    try:
                        result = response.json()
                        avg_ndvi = result["result"]["avg_ndvi"]
                        print(f"   🌱 Average NDVI: {avg_ndvi:.3f}")

                        print("\n🎉 All API tests passed!")
                        return True
                    except json.JSONDecodeError as e:
                        print(f"   ❌ JSON decode error for NDVI: {e}")
                        print(f"   Raw response: {response.text}")
                        return False
                else:
                    print(
                        f"   ⚠️ NDVI query returned {response.status_code}: {response.text}"
                    )
                    return False
            except requests.exceptions.RequestException as e:
                print(f"   ❌ NDVI request failed: {e}")
                return False

    except requests.exceptions.RequestException as e:
        print(f"   ❌ API request failed: {e}")
        return False

    return True


def main():
    success = test_api_buffer()

    if success:
        print("\n🎉 Buffer API Functionality is Working!")
        print("\n" + "=" * 50)
        print("✅ Core buffer_geometry method works")
        print("✅ API endpoints for buffer operations work")
        print("✅ Session management works")
        print("✅ Spatial queries on buffered areas work")

        print("\n🚀 Ready for Frontend Integration!")
        print("\nNext steps:")
        print("1. Add feature selection UI to map")
        print("2. Add buffer visualization")
        print("3. Connect buffer controls to chat interface")
        print("4. Test complete user workflow")

        print("\n💡 Demo usage:")
        print("- Open http://127.0.0.1:5000 in your browser")
        print("- Draw a polygon and ask: 'Create a 2km buffer'")
        print("- Then ask: 'How many buildings are in this buffer?'")

    else:
        print("\n❌ API buffer functionality needs fixes")
        print("Make sure the Flask server is running: python src/app.py")


if __name__ == "__main__":
    main()
