#!/usr/bin/env bash
#
# share.sh — expose the local webapp on a temporary public URL via a
# Cloudflare "quick tunnel" (no Cloudflare account required).
#
# Usage:
#   ./share.sh           # tunnels http://localhost:8501 (this repo's Streamlit app)
#   ./share.sh 7860      # tunnels a different port (e.g. Gradio)
#
# Press Ctrl-C to stop the tunnel.

set -euo pipefail

PORT="${1:-8501}"
TUNNEL_PID=""

cleanup() {
  if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    echo ""
    echo "Shutting down cloudflared (pid ${TUNNEL_PID})…"
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 1) cloudflared must be installed.
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared is not installed." >&2
  echo "Install it with: brew install cloudflared" >&2
  exit 1
fi

# 2) Verify something is actually listening on the port (don't tunnel to nothing).
if ! lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: nothing is listening on http://localhost:${PORT}" >&2
  echo "" >&2
  echo "Start your webapp first, for example:" >&2
  echo "  python -m streamlit run app.py --server.port ${PORT}" >&2
  echo "" >&2
  echo "Or pass the correct port:  ./share.sh <port>" >&2
  exit 1
fi

echo "Starting Cloudflare quick tunnel for http://localhost:${PORT} …"
echo "(this can take a few seconds)"
echo ""

# 3) Run cloudflared, streaming its output to a temp log we also parse for the URL.
LOG_FILE="$(mktemp -t cloudflared_share.XXXXXX)"
cloudflared tunnel --url "http://localhost:${PORT}" >"${LOG_FILE}" 2>&1 &
TUNNEL_PID=$!

# 4) Wait for the public *.trycloudflare.com URL to appear, then print it boxed.
PUBLIC_URL=""
for _ in $(seq 1 30); do
  if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    echo "ERROR: cloudflared exited before producing a URL. Log:" >&2
    cat "${LOG_FILE}" >&2
    exit 1
  fi
  PUBLIC_URL="$(grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "${LOG_FILE}" | head -n 1 || true)"
  [[ -n "${PUBLIC_URL}" ]] && break
  sleep 1
done

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "ERROR: timed out waiting for the tunnel URL. Log:" >&2
  cat "${LOG_FILE}" >&2
  exit 1
fi

BORDER="============================================================"
echo ""
echo "${BORDER}"
echo "  PUBLIC DEMO URL (share this with your advisor):"
echo ""
echo "      ${PUBLIC_URL}"
echo ""
echo "  Serving:  http://localhost:${PORT}"
echo "  Note:     temporary URL — keep this terminal open and your"
echo "            laptop awake. Press Ctrl-C to stop sharing."
echo "${BORDER}"
echo ""

# 5) Keep the tunnel running (and keep streaming cloudflared's log) until Ctrl-C.
tail -f "${LOG_FILE}" &
TAIL_PID=$!
wait "${TUNNEL_PID}"
kill "${TAIL_PID}" 2>/dev/null || true
