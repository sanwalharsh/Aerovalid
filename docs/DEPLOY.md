# Deploying the dashboard to Render

This gets the Streamlit dashboard (`fsa dashboard`) running at a public `https://
<your-service>.onrender.com` URL, password-protected, via Render's Docker-based web
services. It reuses the same `Dockerfile` the CLI/local Docker instructions in
`README.md` use, through `docker/entrypoint.sh`'s `serve-dashboard` mode.

## What this deploys

- A **demo-only** instance: on first boot (and after every restart, since the free
  plan has no persistent disk — see `render.yaml`), the container seeds a small
  2-flight/3-variant run if `runs/` is empty, then serves the dashboard against it.
  It is not a live pipeline-runner connected to your own data; see
  `docs/USER_GUIDE.md` for running studies against real data locally or in your own
  container.
- **Password-gated**: the dashboard requires a password (set as a Render secret,
  never committed) before showing anything, via `AEROVALID_DASHBOARD_PASSWORD`.
- **No other auth, no rate limiting, no HTTPS config needed** — Render terminates
  TLS for you on the `onrender.com` domain automatically.

## One-time setup (steps only you can do — they need your own accounts)

### 1. Push this repo to GitHub

Render's blueprint flow connects to a GitHub (or GitLab) repository. If this repo
isn't on GitHub yet:

```bash
# Create an empty repo on github.com first (via their UI), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin master
git push origin v1.0.0   # optional: push the release tag too
```

### 2. Connect it on Render

1. Sign in at [render.com](https://render.com) (or create an account).
2. **New +** → **Blueprint**.
3. Connect your GitHub account if you haven't, then select this repository.
4. Render detects `render.yaml` at the repo root and shows the `aerovalid-dashboard`
   web service it defines. Click through to create it.
5. Before (or right after) the first deploy, open the service's **Environment**
   tab and add:
   - `AEROVALID_DASHBOARD_PASSWORD` = *(a password of your choice)*
6. Render builds the Docker image and starts the service. The first build takes a
   few minutes (installing dependencies); after that, each boot seeds the demo run
   (well under a minute at this scale) and starts Streamlit.
7. Your dashboard is live at the URL Render shows on the service page
   (`https://aerovalid-dashboard-<random>.onrender.com` by default, or your own
   custom name).

## Updating the deployed version

Render auto-deploys on every push to the connected branch by default. To ship a
change:

```bash
git push origin master
```

## Persistence (optional, needs a paid plan)

The committed `render.yaml` uses Render's free plan, which doesn't support
persistent disks — the demo reseeds on every restart. If you want runs to survive
restarts (e.g. to let the container run a bigger study once and keep showing it),
upgrade the service's plan in the Render dashboard and add a disk block to
`render.yaml` (the exact snippet is in that file's own comments), then redeploy.

## Troubleshooting

- **Blank page / "service unhealthy" in Render's dashboard**: check the service's
  **Logs** tab. The most common cause is the demo-seed step (`fsa run-all`) still
  running — give it a minute on first boot. If it's genuinely stuck, the logs will
  show which pipeline step it's on.
- **"Incorrect password" even though you set it correctly**: confirm the env var
  name is exactly `AEROVALID_DASHBOARD_PASSWORD` (case-sensitive) and that you
  redeployed after setting it (env var changes need a redeploy to take effect).
- **Want to run this somewhere other than Render**: `docker/entrypoint.sh serve-
  dashboard` is host-agnostic — it just needs a `$PORT` env var and a container
  runtime. The same image works on Railway, Fly.io, or a plain VM running Docker;
  only the one-time account-setup steps above differ.
