const express = require('express');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const port = process.env.PORT || 4200;

// Construct the absolute path to replace_base_url.sh
const replaceBaseUrlScript = path.join(__dirname, 'replace_base_url.sh');

// Run the replace_base_url.sh script
exec(replaceBaseUrlScript, (err, stdout, stderr) => {
  console.log('Running replace_base_url.sh from server.js');
  //run ls -lia to see the files in the directory given by __dirname
  exec(`ls -lia ${__dirname}`, (err, stdout, stderr) => {
    console.log(`ls -lia ${__dirname}`);
    console.log(`stdout: ${stdout}`);
    console.log(`stderr: ${stderr}`);
  });
  if (err) {
    console.error(`Error executing replace_base_url.sh: ${err}`);
    return;
  }
  console.log(`replace_base_url.sh output: ${stdout}`);
  if (stderr) {
    console.error(`replace_base_url.sh stderr: ${stderr}`);
  }

// Serve the Angular application
app.use(express.static(path.join(__dirname, 'dist/frontend')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/frontend/index.html'));
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
});