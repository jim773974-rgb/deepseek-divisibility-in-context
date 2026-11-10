import json
import os
import urllib.error
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))
key = open(os.path.join(here, ".deepseek_key")).read().strip()

for path in ("/models", "/user/balance"):
    req = urllib.request.Request(
        "https://api.deepseek.com" + path,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(path, r.status, json.dumps(json.load(r), ensure_ascii=False))
    except urllib.error.HTTPError as e:
        print(path, "HTTP", e.code, e.read().decode("utf-8", "replace")[:300])
    except Exception as e:
        print(path, "ERR", type(e).__name__, e)
