# macOS Deployment Draft

This deployment keeps EpiScope private on the Mac and can expose it externally through a temporary Cloudflare Quick Tunnel.

## Topology

```text
External browser
  -> trycloudflare.com quick-tunnel URL
  -> Cloudflare Tunnel
  -> localhost:8080 on the Mac
  -> Caddy reverse proxy
  -> Streamlit UI (default) or FastAPI API
```

All published container ports bind to `127.0.0.1`, so the services are not reachable directly from the LAN or public internet.

## Files

- `docker-compose.yml`: macOS-oriented stack with local-only bindings
- `Caddyfile`: local reverse proxy

## Prepare Environment

1. Copy `episcope/.env.example` to `episcope/.env`.
2. Fill in real secrets in `episcope/.env`.
3. Use uppercase names such as `MONGO_URI`, not the old lowercase variants.

## Start the Stack Manually

From `episcope/`:

```bash
docker compose -f deploy/macos/docker-compose.yml up -d --build
```

Optional tunnel:

```bash
docker compose -f deploy/macos/docker-compose.yml --profile tunnel up -d
```

To see the temporary public URL:

```bash
docker logs -f episcope-cloudflared
```

Look for a `trycloudflare.com` URL in the logs.

## Local URLs

- UI via proxy: `http://127.0.0.1:8080`
- API health: `http://127.0.0.1:8080/health`
- API endpoints through proxy: `http://127.0.0.1:8080/api/...`
- Direct API: `http://127.0.0.1:8000/health`

## Quick Tunnel Notes

- No domain or Cloudflare account configuration is required for the quick-tunnel flow.
- The public URL is temporary and may change whenever you restart `cloudflared`.
- This is useful for ad hoc sharing and testing, not as a stable production endpoint.

## Operational Notes

- Disable system sleep on the Mac.
- Start the stack manually when you want to publish it.
- Rotate any secrets that were previously committed or stored in plaintext.
- When you migrate to Linux later, reuse the same `.env`, proxy pattern, and service split.
