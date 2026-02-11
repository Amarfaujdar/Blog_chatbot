document.addEventListener('DOMContentLoaded', () => {
  // --- Configuration ---
  const API_URL = '/generate';

  // --- UI Elements ---
  const els = {
    topic: document.getElementById('topicInput'),
    audience: document.getElementById('audienceInput'),
    tone: document.getElementById('toneInput'),
    generateBtn: document.getElementById('generateBtn'),
    markdownBody: document.getElementById('markdownContent'),
    previewTitle: document.getElementById('previewTitle'),
    badgeAudience: document.getElementById('badgeAudience'),
    badgeTone: document.getElementById('badgeTone'),
    historyList: document.getElementById('historyList'),
    themeToggle: document.getElementById('themeToggle'),
    clearHistory: document.getElementById('clearHistory')
  };

  // --- Markdown Renderer Setup ---
  // This is the CRITICAL part for Mermaid.js
  const renderer = new marked.Renderer();

  // Override code block rendering
  renderer.code = function (token, language) {
    // Handle "token" object vs "literal" string (Marked 4.3.0 nuance)
    let code = token;
    let lang = language;

    if (typeof token === 'object') {
      code = token.text;
      lang = token.lang;
    }

    // 1. Check for Mermaid Integration
    if (lang === 'mermaid') {
      // CRITICAL: Return a DIV. Mermaid.js will later find this div
      // and replace its content with the SVG.
      // We DO NOT escape html here to preserve the raw syntax for Mermaid parser.
      return `<div class="mermaid">${code}</div>`;
    }

    // 2. Standard Highlight.js Integration for other languages
    const validLang = !!(lang && hljs.getLanguage(lang));
    const highlighted = validLang ? hljs.highlight(code, { language: lang }).value : code;
    return `<pre><code class="hljs ${lang}">${highlighted}</code></pre>`;
  };

  marked.setOptions({
    renderer: renderer,
    pedantic: false,
    gfm: true,
    breaks: false // Keep false to not break code blocks
  });

  // --- State Management ---
  const AppState = {
    history: JSON.parse(localStorage.getItem('lumina_history') || '[]'),

    addHistory(data) {
      this.history.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        data: data
      });
      this.history = this.history.slice(0, 10); // Keep last 10
      localStorage.setItem('lumina_history', JSON.stringify(this.history));
      Render.history();
    },

    clearHistory() {
      this.history = [];
      localStorage.removeItem('lumina_history');
      Render.history();
    }
  };

  // --- Listeners ---
  els.generateBtn.addEventListener('click', async () => {
    const topic = els.topic.value.trim();
    if (!topic) return alert('Please enter a topic');

    setLoading(true);

    try {
      const payload = {
        topic: topic,
        audience: els.audience.value || 'General',
        tone: els.tone.value || 'Technical'
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error('Generation failed');

      const data = await response.json();

      // Render the result
      await Render.blog(data);

      // Save to history
      AppState.addHistory(data);

    } catch (error) {
      console.error(error);
      els.markdownBody.innerHTML = `
                <div class="empty-state" style="color: #ef4444">
                    <h3>Error Generating Content</h3>
                    <p>${error.message}</p>
                </div>
            `;
    } finally {
      setLoading(false);
    }
  });

  els.clearHistory.addEventListener('click', () => AppState.clearHistory());

  els.themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('lumina_theme', next);

    // IMPORTANT: Re-render diagrams for dark mode/light mode if content exists
    // Mermaid needs to re-run to adjust standard colors
    if (els.markdownBody.querySelector('.mermaid')) {
      // Re-run mermaid on existing content
      mermaid.run({ querySelector: '.mermaid' });
    }
  });

  // --- Rendering Logic ---
  const Render = {
    async blog(data) {
      // Update Headers
      els.previewTitle.textContent = data.blog_title || 'Untitled';
      els.badgeAudience.textContent = data.audience || 'General';
      els.badgeTone.textContent = data.tone || 'Standard';

      // Parse Markdown -> HTML
      let html = marked.parse(data.markdown || '');

      // Sanitize (allow standard tags + mermaid divs)
      // We use DOMPurify to prevent XSS, but allowing our mermaid class
      html = DOMPurify.sanitize(html, {
        ADD_TAGS: ['div'],
        ADD_ATTR: ['class']
      });

      els.markdownBody.innerHTML = html;

      // --- MERMAID RENDERING ---
      // 1. Find all encoded mermaid blocks
      const encodedDivs = els.markdownBody.querySelectorAll('.mermaid-encoded');

      for (const div of encodedDivs) {
        const b64 = div.getAttribute('data-code');
        if (b64) {
          try {
            const decoded = atob(b64);

            // VALIDATION STEP: Check if code is valid mermaid
            // If invalid, we just remove the placeholder and don't show anything
            if (await mermaid.parse(decoded)) {
              // Create replacement div
              const newDiv = document.createElement('div');
              newDiv.className = 'mermaid';
              newDiv.textContent = decoded;
              // Replace the encoded placeholder
              div.replaceWith(newDiv);
            } else {
              // Invalid syntax (should be caught by parse, but just in case)
              console.warn("Mermaid validate failed, hiding diagram.");
              div.remove();
            }
          } catch (e) {
            console.error("Failed to decode/parse mermaid block", e);
            // HIDE THE DIAGRAM IF IT FAILS
            div.remove();
          }
        }
      }

      // 2. Run Mermaid on the newly created .mermaid divs
      try {
        await mermaid.run({
          querySelector: '.mermaid',
          suppressErrors: true
        });
      } catch (e) {
        console.error("Mermaid Render Error:", e);
      }
    },

    history() {
      els.historyList.innerHTML = '';
      AppState.history.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
                    <div style="font-weight:600">${item.data.blog_title || item.data.topic || 'Untitled'}</div>
                    <div style="font-size:11px; opacity:0.7">${new Date(item.timestamp).toLocaleDateString()}</div>
                `;
        div.onclick = () => Render.blog(item.data);
        els.historyList.appendChild(div);
      });
    }
  };

  // --- Helpers ---
  function setLoading(isLoading) {
    els.generateBtn.disabled = isLoading;
    els.generateBtn.querySelector('.btn-text').textContent = isLoading ? 'Generating...' : 'Generate Blueprint';
    const loader = els.generateBtn.querySelector('.btn-loader');
    if (isLoading) loader.classList.remove('hidden'); else loader.classList.add('hidden');
  }

  // Init
  const savedTheme = localStorage.getItem('lumina_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  Render.history();

  // Initialize Mermaid globally
  mermaid.initialize({
    startOnLoad: false, // We manually run it
    theme: savedTheme === 'dark' ? 'dark' : 'default',
    securityLevel: 'loose', // Needed to allow click events etc if we wanted them
    fontFamily: '"Outfit", sans-serif'
  });
});