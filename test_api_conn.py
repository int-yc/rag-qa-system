
import urllib.request
import requests

url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
headers = {"Content-Type": "application/json",
           "Authorization": "Bearer your-api-key-here"}
payload = {"model": "doubao-seed-2-1-pro-260915",
           "messages": [{"role": "user", "content": "hi"}],
           "max_tokens": 5}

print("urllib 检测到的系统代理:", urllib.request.getproxies())

for label, proxies in [("① 默认(可能走系统代理)", None),
                       ("② 强制直连(禁用代理)", {"http": None, "https": None})]:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20, proxies=proxies)
        print(f"{label}: status={r.status_code}, resp={r.text[:100]}")
    except Exception as e:
        print(f"{label}: 失败 -> {type(e).__name__}: {e}")
