import requests
import json

API_URL = "http://localhost:8000/predict"

payload = {
    "lat": 14.1,
    "long": 108.2,
    "muc_nuoc": 0.5,
    "month": 8,
    "rolling_mean_7d": 0.48,
    "delta_1d": 0.02,
    "dung_tich": 0.6,
    "q_den": 0.3,
    "q_xa": 0.25
}

if __name__ == "__main__":
    try:
        print(f"Sending POST request to {API_URL}...")
        print("Payload:", json.dumps(payload, indent=2))
        
        response = requests.post(API_URL, json=payload, timeout=5)
        
        print(f"\nStatus Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2))
            print("\nAPI is working perfectly!")
        else:
            print("Error Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n[!] Error: Could not connect to the API. Have you started the uvicorn server?")
        print("Run it with: uvicorn main:app --reload")
