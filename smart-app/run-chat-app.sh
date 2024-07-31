echo "Running chat-app backend"
cd chat-app/backend/target
# Check if API_KEY is set
if [ -z "$API_KEY" ]; then
  echo "Error: API_KEY is not set."
  exit 1
else
  echo "API_KEY is set."
fi

java -jar ai-demo-0.0.1-SNAPSHOT.jar &

echo "Running chat-app frontend"
cd ../../frontend
ng serve &