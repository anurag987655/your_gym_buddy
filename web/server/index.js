const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
const os = require('os');

const { analyzePose } = require('./analyzePose');

const app = express();
const upload = multer({ dest: os.tmpdir() });

const PORT = Number(process.env.PORT || 4000);
const ALLOWED_POSES = ['squat', 'plank', 'downdog', 'tree', 'warrior2', 'goddess'];

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.get('/api/poses', (_req, res) => {
  res.json({ poses: ALLOWED_POSES });
});

app.post('/api/analyze', upload.single('frame'), async (req, res) => {
  const selectedPose = (req.body.selectedPose || '').toLowerCase().trim();

  if (!req.file) {
    res.status(400).json({ error: 'Missing frame image' });
    return;
  }

  if (!ALLOWED_POSES.includes(selectedPose)) {
    await fs.unlink(req.file.path).catch(() => {});
    res.status(400).json({ error: 'Invalid selectedPose' });
    return;
  }

  try {
    const result = await analyzePose({
      imagePath: req.file.path,
      selectedPose
    });
    res.json(result);
  } catch (err) {
    res.status(500).json({
      error: 'Failed to analyze pose',
      detail: err.message
    });
  } finally {
    await fs.unlink(req.file.path).catch(() => {});
  }
});

app.listen(PORT, () => {
  console.log(`Your Gym Buddy API running on http://localhost:${PORT}`);
});
