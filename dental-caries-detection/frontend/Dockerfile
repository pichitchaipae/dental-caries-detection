# syntax=docker/dockerfile:1

# ---- builder: install all deps, type-check, and build the static bundle ----
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# VITE_* values are baked into the bundle at build time (see docker-compose.yml build.args);
# rebuild the image whenever these change.
ARG VITE_FRONTEND_API_BASE_URL
ARG VITE_POLL_INTERVAL_MS
ENV VITE_FRONTEND_API_BASE_URL=$VITE_FRONTEND_API_BASE_URL
ENV VITE_POLL_INTERVAL_MS=$VITE_POLL_INTERVAL_MS
RUN npm run build

# ---- runtime: `vite preview` serves the built bundle (no Nginx, per project-structure.md 11.8) ----
FROM node:22-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
# `vite` and `@vitejs/plugin-react` are regular dependencies (not devDependencies)
# because `vite preview` loads vite.config.ts, which imports the plugin, at runtime.
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/vite.config.ts ./vite.config.ts

# `vite preview` bundles vite.config.ts into a temp .mjs file in /app on every
# start, so the `node` user needs write access to this directory, not just
# read (COPY leaves everything root-owned by default).
RUN chown -R node:node /app

USER node
EXPOSE 3000

CMD ["npm", "run", "preview"]
