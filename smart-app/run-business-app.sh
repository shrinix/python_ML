echo "Running business-app backend"
cd business-app/backend/target
java -jar ai-demo-0.0.1-SNAPSHOT.jar &

echo "Running business-app frontend"
cd ../../frontend
ng serve &

