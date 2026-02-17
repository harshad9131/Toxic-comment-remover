import { useState, useEffect } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html, body, #root {
    height: 100%;
    width: 100%;
  }

  body {
    background-color: #0a1628;
  }

  .app-wrapper {
    min-height: 100vh;
    width: 100%;
    background: linear-gradient(135deg, #0a1628 0%, #112040 40%, #0d1d35 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    font-family: 'DM Sans', sans-serif;
    position: relative;
    overflow: hidden;
  }

  .app-wrapper::before {
    content: '';
    position: absolute;
    top: -200px;
    left: -200px;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(100, 149, 237, 0.08) 0%, transparent 70%);
    pointer-events: none;
  }

  .app-wrapper::after {
    content: '';
    position: absolute;
    bottom: -100px;
    right: -100px;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(70, 130, 200, 0.06) 0%, transparent 70%);
    pointer-events: none;
  }

  .card {
    width: 100%;
    max-width: 660px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(200, 215, 240, 0.1);
    border-radius: 24px;
    padding: 48px;
    backdrop-filter: blur(12px);
    box-shadow:
      0 4px 24px rgba(0, 0, 0, 0.3),
      0 1px 0 rgba(255, 255, 255, 0.05) inset;
    position: relative;
    z-index: 1;
    animation: fadeSlideIn 0.6s ease forwards;
  }

  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .logo-dot {
    width: 10px;
    height: 10px;
    background: #5a9fff;
    border-radius: 50%;
    box-shadow: 0 0 12px rgba(90, 159, 255, 0.6);
    animation: pulse 2.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
  }

  .subtitle-tag {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5a9fff;
    font-family: 'DM Sans', sans-serif;
  }

  h1 {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #f0eeea;
    line-height: 1.2;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
  }

  .header-desc {
    font-size: 14px;
    color: rgba(240, 238, 234, 0.4);
    font-weight: 300;
    margin-bottom: 36px;
    line-height: 1.5;
  }

  .divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200, 215, 240, 0.12), transparent);
    margin-bottom: 36px;
  }

  .textarea-wrapper {
    position: relative;
    margin-bottom: 20px;
  }

  .textarea-label {
    display: block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(240, 238, 234, 0.4);
    margin-bottom: 10px;
  }

  textarea {
    width: 100%;
    padding: 18px 20px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(200, 215, 240, 0.1);
    border-radius: 14px;
    color: #f0eeea;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 300;
    line-height: 1.7;
    resize: none;
    outline: none;
    transition: border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
  }

  textarea::placeholder {
    color: rgba(240, 238, 234, 0.2);
  }

  textarea:focus {
    border-color: rgba(90, 159, 255, 0.4);
    background: rgba(255, 255, 255, 0.06);
    box-shadow: 0 0 0 3px rgba(90, 159, 255, 0.06);
  }

  .char-count {
    position: absolute;
    bottom: 12px;
    right: 16px;
    font-size: 11px;
    color: rgba(240, 238, 234, 0.2);
    pointer-events: none;
  }

  .submit-btn {
    width: 100%;
    padding: 16px 24px;
    background: linear-gradient(135deg, #2d5fa8 0%, #1e4a8a 100%);
    border: 1px solid rgba(90, 159, 255, 0.3);
    border-radius: 14px;
    color: #f0eeea;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    position: relative;
    overflow: hidden;
  }

  .submit-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    transition: left 0.5s ease;
  }

  .submit-btn:hover::before {
    left: 100%;
  }

  .submit-btn:hover {
    background: linear-gradient(135deg, #3668b8 0%, #2457a0 100%);
    border-color: rgba(90, 159, 255, 0.5);
    box-shadow: 0 4px 20px rgba(45, 95, 168, 0.4);
    transform: translateY(-1px);
  }

  .submit-btn:active {
    transform: translateY(0);
    box-shadow: none;
  }

  .submit-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(240, 238, 234, 0.2);
    border-top-color: #f0eeea;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .result-section {
    margin-top: 32px;
    animation: fadeSlideIn 0.4s ease forwards;
  }

  .result-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(240, 238, 234, 0.4);
    margin-bottom: 16px;
  }

  .result-card {
    padding: 24px 28px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    gap: 18px;
    border: 1px solid;
    position: relative;
    overflow: hidden;
  }

  .result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.04;
    background: radial-gradient(circle at 20% 50%, white, transparent 70%);
  }

  .result-card.toxic {
    background: rgba(220, 60, 60, 0.08);
    border-color: rgba(220, 60, 60, 0.25);
  }

  .result-card.positive {
    background: rgba(50, 185, 130, 0.08);
    border-color: rgba(50, 185, 130, 0.25);
  }

  .result-card.neutral {
    background: rgba(150, 170, 220, 0.08);
    border-color: rgba(150, 170, 220, 0.25);
  }

  .result-card.warning {
    background: rgba(220, 150, 50, 0.08);
    border-color: rgba(220, 150, 50, 0.25);
  }

  .result-icon {
    font-size: 32px;
    line-height: 1;
    flex-shrink: 0;
  }

  .result-text-group {
    flex: 1;
  }

  .result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 4px;
  }

  .result-card.toxic .result-verdict { color: #f87575; }
  .result-card.positive .result-verdict { color: #5dd9a8; }
  .result-card.neutral .result-verdict { color: #a8b8e8; }
  .result-card.warning .result-verdict { color: #f0b860; }

  .result-desc {
    font-size: 13px;
    font-weight: 300;
    color: rgba(240, 238, 234, 0.4);
    line-height: 1.5;
  }

  .all-results {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .tag {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
    border: 1px solid;
  }

  .tag.active-toxic {
    background: rgba(220, 60, 60, 0.15);
    border-color: rgba(220, 60, 60, 0.4);
    color: #f87575;
  }

  .tag.inactive {
    background: rgba(255, 255, 255, 0.03);
    border-color: rgba(200, 215, 240, 0.1);
    color: rgba(240, 238, 234, 0.25);
  }

  .error-msg {
    margin-top: 16px;
    padding: 14px 18px;
    background: rgba(220, 60, 60, 0.08);
    border: 1px solid rgba(220, 60, 60, 0.2);
    border-radius: 10px;
    color: #f87575;
    font-size: 13px;
    font-weight: 300;
  }
`;

const LABEL_CONFIG = {
  toxic:            { icon: "⚠️", desc: "This comment contains toxic language" },
  severe_toxic:     { icon: "🚨", desc: "Severely harmful content detected" },
  obscene:          { icon: "🤬", desc: "Obscene language found" },
  threat:           { icon: "🔴", desc: "Threatening language detected" },
  insult:           { icon: "💢", desc: "Insulting content found" },
  identity_hate:    { icon: "⛔", desc: "Hate speech targeting identity groups" },
};

function getOverallVerdict(predictions) {
  if (!predictions) return null;

  const ANY_TOXIC_THRESHOLD = 0.5;
  const triggered = Object.entries(predictions).filter(
    ([key, prob]) => key !== "non_toxic" && prob >= ANY_TOXIC_THRESHOLD
  );

  if (triggered.length === 0) {
    return {
      verdict: "Positive",
      type: "positive",
      icon: "✅",
      desc: "No harmful content detected",
      tags: []
    };
  }

  const topKey = triggered.sort((a, b) => b[1] - a[1])[0][0];
  const cfg = LABEL_CONFIG[topKey] || { icon: "⚠️", desc: "Harmful content detected" };
  const displayName = topKey.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  const cardType = topKey === "severe_toxic" || topKey === "threat"
    ? "toxic"
    : topKey === "identity_hate"
    ? "warning"
    : "toxic";

  return {
    verdict: displayName,
    type: cardType,
    icon: cfg.icon,
    desc: cfg.desc,
    tags: triggered.map(([k]) => k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()))
  };
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const styleEl = document.createElement("style");
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
    return () => document.head.removeChild(styleEl);
  }, []);

  const API_URL = "http://127.0.0.1:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data.predictions);
    } catch {
      setError("Could not reach the backend. Make sure the server is running.");
    }

    setLoading(false);
  };

  const verdict = getOverallVerdict(result);

  return (
    <>
      <div className="app-wrapper">
        <div className="card">
          <div className="logo-row">
            <div className="logo-dot" />
            <span className="subtitle-tag">Content Safety</span>
          </div>

          <h1>Toxicity Detector</h1>
          <p className="header-desc">
            Paste any YouTube comment below to instantly analyse its tone and safety.
          </p>

          <div className="divider" />

          <form onSubmit={handleSubmit}>
            <div className="textarea-wrapper">
              <label className="textarea-label">Comment</label>
              <textarea
                rows={5}
                placeholder="e.g. This video is absolutely amazing, keep it up!"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <span className="char-count">{text.length}</span>
            </div>

            <button
              type="submit"
              className="submit-btn"
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analysing…
                </>
              ) : (
                "Analyse Comment"
              )}
            </button>
          </form>

          {error && <div className="error-msg">{error}</div>}

          {verdict && (
            <div className="result-section">
              <div className="result-label">Analysis Result</div>
              <div className={`result-card ${verdict.type}`}>
                <div className="result-icon">{verdict.icon}</div>
                <div className="result-text-group">
                  <div className="result-verdict">{verdict.verdict}</div>
                  <div className="result-desc">{verdict.desc}</div>
                </div>
              </div>

              {verdict.tags.length > 0 && (
                <div className="all-results">
                  {verdict.tags.map((tag) => (
                    <span key={tag} className="tag active-toxic">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}