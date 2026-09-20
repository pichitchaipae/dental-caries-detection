# syntax=docker/dockerfile:1

# ---- builder: install all deps and compile TypeScript ----
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json tsconfig.build.json ./
COPY src ./src
RUN npm run build

# ---- runtime: production deps + compiled output only ----
FROM node:22-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=builder /app/dist ./dist

# A fresh named volume copies the ownership of this directory on first mount,
# so the non-root user can write input.jpg / result.json into /shared.
RUN mkdir -p /shared && chown node:node /shared

USER node
EXPOSE 8000

# `exec` makes node PID 1 so it receives SIGTERM from `docker compose stop`.
CMD ["sh", "-c", "node dist/db/migrate.js && exec node dist/server.js"]
