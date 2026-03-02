# Web App (React + Node)

## 1) Install dependencies

```bash
cd ~/Desktop/your_gym_buddy/web/server && npm install
cd ~/Desktop/your_gym_buddy/web/client && npm install
```

## 2) Start backend API

```bash
cd ~/Desktop/your_gym_buddy/web/server
npm run dev
```

API runs at `http://localhost:4000`.

## 3) Start frontend

```bash
cd ~/Desktop/your_gym_buddy/web/client
npm run dev
```

Frontend runs at `http://localhost:5173`.

## One-command run (server + client)

```bash
cd ~/Desktop/your_gym_buddy/web
npm run dev
```

## How it works

- Frontend captures webcam frame + selected pose.
- Backend sends the frame to `scripts/web_pose_feedback.py`.
- Python computes pose metrics and returns deterministic coaching feedback.
