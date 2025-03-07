FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the .env file
COPY *.env ./

# Copy the rest of the application code
COPY . .

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Ensure the entrypoint script has execute permissions
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on
EXPOSE 5001

# Use the shell script as the entry point
ENTRYPOINT ["/app/entrypoint.sh"]