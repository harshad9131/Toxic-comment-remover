import { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "http://127.0.0.1:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      setResult(data.predictions);
    } catch (error) {
      console.error("Error:", error);
      alert("Backend not running?");
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "700px", margin: "40px auto", fontFamily: "Arial" }}>
      <h1>YouTube Toxic Comment Detector</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          rows="5"
          style={{ width: "100%", padding: "10px" }}
          placeholder="Paste a YouTube comment here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          type="submit"
          style={{
            marginTop: "10px",
            padding: "10px 20px",
            cursor: "pointer"
          }}
        >
          {loading ? "Checking..." : "Check Toxicity"}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h3>Results:</h3>

          {Object.entries(result).map(([label, prob]) => (
            <div key={label} style={{ marginBottom: "10px" }}>
              <strong>{label}</strong>: {(prob * 100).toFixed(2)}%
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
