# Host FastAPI webapp on Mac Mini via Cloudflare domain

Expose the local uvicorn app (`127.0.0.1:8001`) on your Cloudflare domain using a
**named Cloudflare Tunnel**. No router port-forwarding and no public home IP needed.

**Security note:** this webapp has **no built-in login**. Anyone who can reach the
URL can use it (and trigger WRDS/Refinitiv actions if credentials are present).
Put **Cloudflare Access** in front (email allowlist) unless you intentionally want
it public.

Replace `app.example.com` and paths with your values.

---

## 0. Prerequisites (Mac Mini)

- Domain already on Cloudflare (nameservers pointing at Cloudflare).
- Project running locally:

```bash
cd /path/to/Sentiment_learn_to_rank_paper
conda activate sentiment-ltr-paper
caffeinate -is python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8001
```

Confirm: http://127.0.0.1:8001 returns 200.

Install the tunnel client:

```bash
brew install cloudflared
cloudflared --version
```

---

## 1. Authenticate cloudflared (once)

```bash
cloudflared tunnel login
```

A browser opens → pick the Cloudflare account → authorize the domain.
This writes a cert under `~/.cloudflared/`.

---

## 2. Create a named tunnel

```bash
cloudflared tunnel create sentiment-ltr
```

Note the tunnel **UUID** printed (also listed by `cloudflared tunnel list`).
A credentials JSON lands in `~/.cloudflared/<UUID>.json`.

---

## 3. Config file

```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Use (edit UUID + hostname):

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /Users/<YOUR_MAC_USER>/.cloudflared/<TUNNEL_UUID>.json

ingress:
  - hostname: app.example.com
    service: http://127.0.0.1:8001
  - service: http_status:404
```

Validate:

```bash
cloudflared tunnel ingress validate
```

---

## 4. Point DNS at the tunnel

```bash
cloudflared tunnel route dns sentiment-ltr app.example.com
```

In Cloudflare Dashboard → DNS you should see a **CNAME**  
`app` → `<TUNNEL_UUID>.cfargotunnel.com` (proxied / orange cloud).

---

## 5. Run the tunnel

Foreground test:

```bash
cloudflared tunnel run sentiment-ltr
```

Open `https://app.example.com` (HTTPS is terminated by Cloudflare).

Install as a LaunchAgent so it survives reboot:

```bash
sudo cloudflared service install
# or user-level:
cloudflared service install
```

Check:

```bash
sudo launchctl list | grep cloudflared || launchctl list | grep -i cloud
cloudflared tunnel info sentiment-ltr
```

Keep uvicorn up separately (tmux / `caffeinate` / LaunchAgent). The tunnel only
forwards; if uvicorn is down, the domain shows an error.

---

## 6. Cloudflare Access (recommended)

Dashboard → **Zero Trust** → **Access** → **Applications** → Add application:

- Type: Self-hosted  
- Application domain: `app.example.com`  
- Policy: Allow emails = your address (and any collaborators)

Visitors hit an email/OTP (or IdP) gate before the FastAPI UI.

Optional: under the hostname, enable **WAF** / Bot Fight if you leave it more open.

---

## 7. Keep uvicorn always on (Mini)

Example LaunchAgent `~/Library/LaunchAgents/com.sentimentltr.uvicorn.plist`
(adjust paths):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.sentimentltr.uvicorn</string>
  <key>WorkingDirectory</key>
  <string>/Users/YOU/path/to/Sentiment_learn_to_rank_paper</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-is</string>
    <string>/Users/YOU/miniconda/envs/sentiment-ltr-paper/bin/python</string>
    <string>-m</string>
    <string>uvicorn</string>
    <string>webapp.main:app</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8001</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/YOU/Library/Logs/sentimentltr-uvicorn.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOU/Library/Logs/sentimentltr-uvicorn.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.sentimentltr.uvicorn.plist
```

Bind **only** `127.0.0.1` when using a tunnel (do not use `0.0.0.0` on the LAN
unless you also want local-network access).

---

## Quick vs named tunnel

| Method | URL | Persistence |
| --- | --- | --- |
| `./share.sh 8001` | random `*.trycloudflare.com` | dies when process stops |
| Named tunnel (this doc) | your domain | survives with LaunchAgent |

---

## Checklist

- [ ] uvicorn on `127.0.0.1:8001`
- [ ] `cloudflared tunnel login` + `create` + `config.yml`
- [ ] DNS CNAME via `tunnel route dns`
- [ ] `https://app.example.com` loads `/batch`
- [ ] Cloudflare Access policy (or accept public exposure)
- [ ] Both uvicorn + cloudflared set to restart on login/reboot

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| 502 Bad Gateway | uvicorn not running; check localhost:8001 |
| DNS not resolving | wait a minute; confirm orange-cloud CNAME in Cloudflare DNS |
| Login loop / Access | Zero Trust policy email mismatch |
| Works on LAN only | you used `0.0.0.0` without tunnel; use tunnel + `127.0.0.1` |
| Duo/WRDS from remote browser | WRDS still runs **on the Mini**; MFA is for Mini IP, not visitor IP |
