# hospital/ — the benchmark server

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

The endpoints used: `/env/reset`, `/env/step` (one episode, filtered so the answer key is
never shown) and `/score` (needs the scorer token, which only our code holds).

First boot loads the database and downloads ICD-10-CM titles from CMS (a few minutes).
The download matters: the reward's chart-neutral rule matches against those titles, so
`python -m audit.verify` checks the table is populated.
