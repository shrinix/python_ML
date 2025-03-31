const express = require('express');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const port = process.env.PORT || 4200;

// Construct the absolute path to replace_urls.sh
const replaceUrlsScript = path.join(__dirname, 'replace_urls.sh');


// Parse command-line arguments
const args = process.argv.slice(2);
const replaceUrlsExecuted = args.includes('--replace-urls-executed=true');

if (replaceUrlsExecuted) {
  console.log('replace_urls.sh has already been executed. Skipping related tasks.');
} else {
  console.log('replace_urls.sh has not been executed. Proceeding with related tasks.');
  // Run the replace_urls.sh script
  exec(replaceUrlsScript, (err, stdout, stderr) => {
    console.log('Running replace_urls.sh from server.js');
    //run ls -lia to see the files in the directory given by __dirname
    exec(`ls -lia ${__dirname}`, (err, stdout, stderr) => {
      console.log(`ls -lia ${__dirname}`);
      console.log(`stdout: ${stdout}`);
      console.log(`stderr: ${stderr}`);
    });
    if (err) {
      console.error(`Error executing replace_urls.sh: ${err}`);
      return;
    }
    console.log(`replace_urls.sh output: ${stdout}`);
    if (stderr) {
      console.error(`replace_urls.sh stderr: ${stderr}`);
    }
  });
  // Set the environment variable to indicate that replace_urls.sh has been executed
  process.env.REPLACE_URLS_EXECUTED = 'true';
}

console.log('Current working directory:', __dirname);
// Serve the Angular application
app.use(express.static(path.join(__dirname, 'dist/frontend')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/frontend/index.html'));
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
