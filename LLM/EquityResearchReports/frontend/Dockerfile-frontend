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

#All the copy and run commands should be under thw WORKDIR command
WORKDIR /app

#Copy env files to the container
COPY --from=build /app/src/environments/environment.prod.ts /app/src/environments/environment.prod.ts
COPY --from=build /app/src/environments/environment.template.ts /app/src/environments/environment.template.ts
COPY --from=build /app/src/environments/environment.ts /app/src/environments/environment.ts

COPY --from=build /app/dist/frontend /app/dist/frontend
COPY --from=build /app/node_modules /app/node_modules
COPY --from=build /app/package.json /app/package.json
COPY server.js /app/server.js
COPY replace_base_url.sh /app/replace_base_url.sh
# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Ensure the script has execute permissions
RUN chmod +x /app/replace_base_url.sh
# Ensure the entrypoint script has execute permissions
RUN chmod +x /app/entrypoint.sh

RUN npm install express

EXPOSE 4200

CMD ["/app/entrypoint.sh"]