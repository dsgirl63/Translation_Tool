"""
Quick test script to verify the Flask app works locally
Run this before deploying: python test_local.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the Flask app is running: python app.py")
        return False

def test_translation(text, language):
    """Test translation endpoint"""
    print(f"\n🔍 Testing translation: '{text}' → {language}...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/translate",
            json={"text": text, "language": language},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Translation successful!")
            print(f"   Input: {text}")
            print(f"   Output: {data.get('translation', 'N/A')}")
            return True
        else:
            print(f"❌ Translation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return False

def main():
    print("=" * 50)
    print("🧪 Testing AI Language Translator API")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Please start the server first.")
        return
    
    # Wait a bit for models to load if needed
    print("\n⏳ Waiting for models to be ready...")
    time.sleep(2)
    
    # Test translations
    print("\n" + "=" * 50)
    print("Testing Translations")
    print("=" * 50)
    
    test_cases = [
        ("Hello, how are you?", "French"),
        ("Good morning", "Spanish"),
        ("Thank you very much", "French"),
    ]
    
    results = []
    for text, lang in test_cases:
        result = test_translation(text, lang)
        results.append(result)
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ 'requests' library not found. Install it with: pip install requests")
        exit(1)
    
    main()
