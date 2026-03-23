// ══════════════════════════════════════════════════════════════════════════
// Blood Cell Anomaly Detection — Frontend Logic
// ══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const btn = document.getElementById('btn-predict');
  const resultPanel = document.getElementById('result-panel');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI: loading state
    btn.classList.add('loading');
    btn.disabled = true;
    resultPanel.classList.remove('visible');

    try {
      const formData = new FormData(form);

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        renderResult(data.result);
      } else {
        showError(data.error || 'Bilinmeyen hata oluştu.');
      }
    } catch (err) {
      showError('Sunucuya bağlanılamadı: ' + err.message);
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  });

  function renderResult(result) {
    // Badge
    const badge = document.getElementById('result-badge');
    badge.className = 'result-badge ' + result.severity;
    const severityText = {
      success: '✅ Normal',
      warning: '⚠️ Dikkat',
      danger: '🚨 Yüksek Risk',
      info: 'ℹ️ Bilgi'
    };
    badge.textContent = severityText[result.severity] || result.severity;

    // Label
    document.getElementById('result-label').textContent = result.label;
    document.getElementById('result-label-tr').textContent = result.label_tr;

    // Confidence
    const confEl = document.getElementById('result-confidence');
    confEl.textContent = result.confidence.toFixed(1) + '%';
    confEl.className = 'result-confidence ' + result.severity;

    // Probability bars
    const probGrid = document.getElementById('prob-grid');
    probGrid.innerHTML = '';

    // Sort by probability descending
    const sorted = Object.entries(result.probabilities)
      .sort((a, b) => b[1] - a[1]);

    sorted.forEach(([name, prob], idx) => {
      const item = document.createElement('div');
      item.className = 'prob-item';

      const nameEl = document.createElement('div');
      nameEl.className = 'prob-name';
      nameEl.textContent = name.replace(/_/g, ' ');

      const barOuter = document.createElement('div');
      barOuter.className = 'prob-bar-outer';

      const barInner = document.createElement('div');
      barInner.className = 'prob-bar-inner' + (idx === 0 ? ' top' : '');
      barInner.style.width = '0%';

      barOuter.appendChild(barInner);

      const valEl = document.createElement('div');
      valEl.className = 'prob-value';
      valEl.textContent = prob.toFixed(1) + '%';

      item.appendChild(nameEl);
      item.appendChild(barOuter);
      item.appendChild(valEl);
      probGrid.appendChild(item);

      // Animate bar after a tiny delay
      requestAnimationFrame(() => {
        setTimeout(() => {
          barInner.style.width = Math.max(prob, 0.5) + '%';
        }, 80 + idx * 60);
      });
    });

    // Show panel
    resultPanel.classList.add('visible');

    // Scroll to result
    setTimeout(() => {
      resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
  }

  function showError(message) {
    const badge = document.getElementById('result-badge');
    badge.className = 'result-badge danger';
    badge.textContent = '❌ Hata';

    document.getElementById('result-label').textContent = 'Tahmin Başarısız';
    document.getElementById('result-label-tr').textContent = message;

    const confEl = document.getElementById('result-confidence');
    confEl.textContent = '—';
    confEl.className = 'result-confidence';

    document.getElementById('prob-grid').innerHTML = '';

    resultPanel.classList.add('visible');
  }
});
