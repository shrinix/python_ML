# Stage 1: Build the Angular application
FROM node:18 AS build

WORKDIR /app

COPY package.json package-lock.json ./

# Install dependencies
RUN npm install
RUN npm install -g @angular/cli

COPY . .

# Build the Angular application
RUN npm run build

# Stage 2: Serve the Angular application using Node.js
FROM node:18

WORKDIR /app

COPY --from=build /app/dist/frontend /app/dist/frontend
COPY server.js /app/server.js

RUN npm install express

EXPOSE 4200

CMD ["node", "server.js"]