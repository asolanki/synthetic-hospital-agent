# hospital/

[Synthetic Hospital](https://github.com/sparkcpark/synthetic_hospital) v1.3 is a web server
holding 1,268 synthetic patients, their answer keys, and the official scorer. Every other
folder talks to it over HTTP on `http://localhost:58000`; the model never sees the database.

```bash
hospital/setup.sh                                    # clone upstream at the pinned commit, build, start
docker compose -f hospital/compose.yaml ps           # app, postgres, redis
docker compose -f hospital/compose.yaml logs -f app  # first boot: "database ready: 1268 patients"
docker compose -f hospital/compose.yaml down         # stop (keeps the database volume)
```

| File | What it is |
|---|---|
| `setup.sh` | Clones upstream into `external/synthetic_hospital/` (gitignored) at a fixed commit, starts the stack, waits for health |
| `compose.yaml` | Three containers: the app (database baked into the image), Postgres, Redis |
| `setup_native.sh` | Ubuntu 24.04 GPU-container setup: pinned upstream, separate Python environment, native databases and Supervisor services |
| `run_native.sh` | Private, four-worker hospital API used by Supervisor |

The endpoints used: `/env/reset`, `/env/step` (one episode, filtered so the answer key is
never shown) and `/score` (needs the scorer token, which only our code holds).

First boot loads the database and downloads ICD-10-CM titles from CMS (a few minutes).
The download matters: the reward's chart-neutral rule matches against those titles, so
`python -m audit.verify` checks the table is populated.

## GPU containers without Docker

On a fresh Ubuntu 24.04 CUDA container with `uv` and Supervisor, run as root:

```bash
bash hospital/setup_native.sh
supervisorctl status shrl-postgres shrl-redis shrl-hospital
set -a; source .env; set +a
curl -f http://127.0.0.1:58000/health
```

This uses the same upstream commit and database loaders as the Docker setup.
Postgres 16, Redis 7, and the API listen only on loopback. The script creates a
host-local `.env` with random database, JWT, and scorer secrets if it does not
already exist. Do not upload your workstation `.env` or account-wide API keys.
The simulator dependencies live in `.venv-hospital`; Verifiers stays in `.venv`.
Logs are in `runs/services/`.

The native deployment was exercised on eight dev patients: all oracle answers
scored 1.0, an empty answer scored 0.0, and 97,584 ICD-10-CM rows were loaded.
This is a deployment smoke check, not a rerun of the entire task audit.
The existing `audit.verify` terminology check uses Docker; it is not the native
deployment health command.

On Vast, stopping retains the filesystem but still incurs storage charges.
Destroying/recycling a container without a mounted volume loses its data.
Copy results and checkpoints off the instance before destroying it.
