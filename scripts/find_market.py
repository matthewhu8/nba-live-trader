import requests
resp = requests.get('https://external-api.kalshi.com/trade-api/v2/markets?ticker=KXNBASPREAD-26MAY06MINSAS')
if resp.status_code == 200:
    print(resp.json())
else:
    print("Failed", resp.status_code)
