const state = { sample: null };
const $ = (id) => document.getElementById(id);

function drawLine(canvas, series, colors, residual = false) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * ratio); canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); const width = rect.width; const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const values = residual ? series : series.flat(); const limit = residual ? Math.max(...values.map(Math.abs), .01) : 1.05;
  series.forEach((line, index) => { ctx.beginPath(); ctx.lineWidth = index === 0 && !residual ? 1.2 : 1.7; ctx.strokeStyle = colors[index]; line.forEach((value, point) => { const x = point / (line.length - 1) * width; const y = height / 2 - (value / limit) * (height * .43); point ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); });
}

function render(sample) {
  state.sample = sample; $('sample-index').value = sample.index; $('sample-total').textContent = `/ ${sample.total_samples - 1}`; $('record').textContent = sample.record; $('offset').textContent = `${sample.start} samples`; $('duration').textContent = `${(sample.noisy.length / sample.sample_rate_hz * 1000).toFixed(1)} ms`; $('snr-badge').textContent = `SNR ${sample.target_snr_db.toFixed(1)} dB`; $('input-prd').textContent = sample.metrics.input_prd.toFixed(2); $('output-prd').textContent = sample.metrics.output_prd.toFixed(2); $('mse').textContent = sample.metrics.mse.toFixed(5); $('loading').textContent = 'Window loaded';
  drawLine($('waveform'), [sample.noisy, sample.denoised, sample.clean], ['#dc7644', '#168477', '#c1a354']); drawLine($('residual'), [sample.residual], ['#dc7644'], true);
}

async function loadSample(index) { $('loading').textContent = 'Running inference…'; const response = await fetch(`/api/sample?index=${index}`); const payload = await response.json(); if (!response.ok) throw new Error(payload.error); render(payload); }
async function updateTrainingProgress() { try { const response = await fetch('/api/training-progress'); const progress = await response.json(); const completed = progress.completed_epochs || 0; const total = progress.total_epochs || 60; const percent = Math.min(100, completed / total * 100); $('training-bar').style.width = `${percent}%`; $('training-count').textContent = `${completed} / ${total} epochs`; $('training-status').textContent = progress.status === 'completed' ? 'Training complete' : progress.status === 'early_stopped' ? 'Early stopped' : completed ? `Epoch ${completed} complete` : 'Starting training'; } catch (error) { $('training-status').textContent = 'Progress unavailable'; } }
async function boot() { try { const health = await (await fetch('/api/health')).json(); $('status-text').textContent = `Model online · ${health.samples} windows`; $('status').classList.add('ready'); await loadSample(0); updateTrainingProgress(); setInterval(updateTrainingProgress, 3000); } catch (error) { $('status-text').textContent = 'Could not load model'; $('loading').textContent = error.message; } }
$('previous').onclick = () => loadSample(Math.max(0, state.sample.index - 1)); $('next').onclick = () => loadSample(Math.min(state.sample.total_samples - 1, state.sample.index + 1)); $('sample-index').onchange = (event) => loadSample(Number(event.target.value)); window.onresize = () => state.sample && render(state.sample); boot();