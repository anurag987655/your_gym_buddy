const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

function analyzePose({ imagePath, selectedPose }) {
  return new Promise((resolve, reject) => {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'web_pose_feedback.py');
    const venvPython = path.join(projectRoot, 'venv', 'bin', 'python');
    const pythonBin = fs.existsSync(venvPython) ? venvPython : 'python3';

    const py = spawn(pythonBin, [scriptPath, '--image', imagePath, '--pose', selectedPose], {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    const timeoutId = setTimeout(() => {
      py.kill('SIGKILL');
      reject(new Error('Python analyzer timed out'));
    }, 20000);

    py.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    py.on('error', (err) => {
      clearTimeout(timeoutId);
      reject(err);
    });

    py.on('close', (code) => {
      clearTimeout(timeoutId);
      if (code !== 0) {
        reject(new Error(`Analyzer failed (${code}): ${stderr || stdout}`));
        return;
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Invalid analyzer output: ${stdout || stderr}`));
      }
    });
  });
}

module.exports = { analyzePose };
