echo "Running business-app"
./run-business-app.sh &
./run-chat-app.sh &
# Wait for all background processes to finish
wait
