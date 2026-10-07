# ──────────────────────────────────────────────────────────────
#  Frontend — React + Vite → served by nginx
#  Multi-stage build: Stage 1 builds, Stage 2 serves via nginx
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Build React app ──────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package manifests and install dependencies first (layer cache)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build production bundle
COPY . .
RUN npm run build

# ── Stage 2: Serve with nginx ─────────────────────────────────
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Replace default nginx config with our custom proxy config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
