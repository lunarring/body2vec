import requests

def post_request():
    url = "https://simple-api.glif.app/clrtepmrr0006mmvdij15pnn7"
    headers = {
        "Authorization": "Bearer YOUAPIKEY"
    }
    data = {
        "input": ["a cute rabbit", "dark horror"]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Call the function
response = post_request()
print(response)