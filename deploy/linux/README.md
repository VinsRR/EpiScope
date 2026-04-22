# Linux Deployment Draft

This deployment is intended for a Linux server with Docker and Docker Compose available. It keeps the API, UI, and Qdrant private on the Docker network and exposes only a single reverse proxy on port `80`.

## Topology

```text
External browser
  -> http://SERVER_IP/
  -> Caddy reverse proxy on port 80
  -> Streamlit UI (default) or FastAPI API
```

API traffic is exposed through the same proxy under `/api/...` and `/health`.

## Files

- `docker-compose.yml`: Linux-oriented stack for a public server
- `Caddyfile`: reverse proxy published on port `80`
- `episcope-stack.service`: optional `systemd` unit for automatic startup

## Prepare Environment

1. Copy `episcope/.env.example` to `episcope/.env`.
2. Fill in the real runtime secrets in `episcope/.env`.
3. Keep using uppercase names such as `MONGO_URI`.

## Start the Stack Manually

From `episcope/`:

```bash
docker compose -f deploy/linux/docker-compose.yml up -d --build
```

## Public URLs

- UI: `http://SERVER_IP/`
- API health: `http://SERVER_IP/health`
- API endpoints: `http://SERVER_IP/api/...`

## Firewall Notes

- Open inbound TCP port `80`.
- The API, UI, and Qdrant containers are not published directly to the host.

## Optional systemd Startup

If you want the stack to start automatically on boot:

1. Deploy the repository to a stable path such as `/opt/episcope`.
2. Adjust paths in `episcope-stack.service` if needed.
3. Copy the unit:

```bash
sudo cp deploy/linux/episcope-stack.service /etc/systemd/system/episcope-stack.service
```

4. Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now episcope-stack.service
```

5. Check status:

```bash
sudo systemctl status episcope-stack.service
```

## Notes

- This draft serves plain HTTP by default. It is a good first deployment for a private server or for use behind another TLS terminator.
- If you later add a real domain, you can switch the proxy layer to domain-based HTTPS without changing the API/UI split.
- The macOS deployment in `deploy/macos/` is unchanged.
