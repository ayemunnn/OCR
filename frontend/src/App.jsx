import React, { useEffect, useState } from "react";

import { getCurrentUser } from "./api/client";
import DocumentList from "./components/DocumentList";
import DocumentUpload from "./components/DocumentUpload";
import LoginForm from "./components/LoginForm";
import SignupForm from "./components/SignupForm";

const pageStyle = {
  maxWidth: "920px",
  margin: "0 auto",
  padding: "32px 20px",
  fontFamily: "Arial, sans-serif",
};

const panelStyle = {
  border: "1px solid #ddd",
  borderRadius: "8px",
  padding: "20px",
  marginBottom: "20px",
};

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("papersleuth_token"));
  const [currentUser, setCurrentUser] = useState(null);
  const [refreshDocuments, setRefreshDocuments] = useState(0);
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function loadCurrentUser() {
      if (!token) {
        setCurrentUser(null);
        return;
      }

      try {
        const user = await getCurrentUser(token);
        setCurrentUser(user);
      } catch (error) {
        localStorage.removeItem("papersleuth_token");
        setToken(null);
        setCurrentUser(null);
        setMessage(error.message);
      }
    }

    loadCurrentUser();
  }, [token]);

  function handleLogin(accessToken) {
    localStorage.setItem("papersleuth_token", accessToken);
    setToken(accessToken);
    setMessage("Logged in successfully.");
  }

  function handleLogout() {
    localStorage.removeItem("papersleuth_token");
    setToken(null);
    setCurrentUser(null);
    setMessage("Logged out.");
  }

  return (
    <main style={pageStyle}>
      <h1>PaperSleuth</h1>
      <p>Simple frontend shell for the PaperSleuth FastAPI backend.</p>

      {message && <p>{message}</p>}

      {!token ? (
        <section style={{ display: "grid", gap: "20px", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
          <div style={panelStyle}>
            <SignupForm onSignup={setMessage} />
          </div>
          <div style={panelStyle}>
            <LoginForm onLogin={handleLogin} />
          </div>
        </section>
      ) : (
        <>
          <section style={panelStyle}>
            <p>
              Logged in as <strong>{currentUser?.email || "loading..."}</strong>
            </p>
            <button type="button" onClick={handleLogout}>
              Log out
            </button>
          </section>

          <section style={panelStyle}>
            <DocumentUpload
              token={token}
              onUploaded={() => setRefreshDocuments((value) => value + 1)}
            />
          </section>

          <section style={panelStyle}>
            <DocumentList token={token} refreshKey={refreshDocuments} />
          </section>
        </>
      )}
    </main>
  );
}

export default App;
