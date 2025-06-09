import requests

API_KEY = "AIzaSyCvv3R2E4iIVpJ4pRhgpzGdNuFp7Lp2lcw"

def test_api_key(api_key):
    # We'll query a Google public API metadata endpoint that lists enabled services for a project
    url = f"https://www.googleapis.com/discovery/v1/apis/customsearch/v1/rest"
    
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"key": api_key})
        if response.status_code == 200:
            print("✅ API key is valid and Custom Search API is accessible.")
        elif response.status_code == 403:
            print("❌ API key is valid but access to Custom Search API is forbidden.")
            print(f"Details: {response.json()}")
        else:
            print(f"❌ Unexpected response code: {response.status_code}")
            print(f"Details: {response.text}")
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")

if __name__ == "__main__":
    print("🔍 Testing API key access to Google Custom Search API...")
    test_api_key(API_KEY)
