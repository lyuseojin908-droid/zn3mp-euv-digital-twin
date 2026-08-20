# Zn–3MP EUV Process Digital Twin — React + Netlify

A portfolio-style React dashboard that reproduces the Gaussian Process Regression surrogate model entirely in the browser.

## Features
- React + Vite
- Pure client-side GPR prediction
- Interactive temperature / hfacH pressure / Al₂O₃ controls
- Process field + uncertainty field
- UCB-based experiment planner
- Netlify-ready configuration
- No Python server required

## Local run
```bash
npm install
npm run dev
```

## Netlify deployment
1. Upload this project to a GitHub repository.
2. In Netlify: **Add new project → Import an existing project → GitHub**.
3. Select the repository.
4. Build command: `npm run build`
5. Publish directory: `dist`
6. Deploy.

`netlify.toml` already contains the correct build settings.

## Research note
Some training points are approximate values digitized from Figure 2. Model outputs are intended for process exploration and follow-up experiment prioritization, not for claiming experimentally verified optima.
