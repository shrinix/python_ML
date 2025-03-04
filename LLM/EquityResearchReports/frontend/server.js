const express = require('express');
const path = require('path');

const app = express();

const PORT = process.env.PORT || 4200;

// Serve static files from the Angular app
app.use(express.static(path.join(__dirname, 'dist/frontend')));

// Catch all other routes and return the index file
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/frontend/index.html'));
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});