import React, { useEffect, useState } from "react";

import { getCurrentUser } from "./api/client";
import DocumentList from "./components/DocumentList";
import DocumentUpload from "./components/DocumentUpload";
import LoginForm from "./components/LoginForm";
import SignupForm from "./components/SignupForm";
import DocumentDetail from "./components/DocumentDetail";

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("papersleuth_token"));
  const [currentUser, setCurrentUser] = useState(null);
  const [refreshDocuments, setRefreshDocuments] = useState(0);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [message, setMessage] = useState("");
  const [authMode, setAuthMode] = useState("login");
  const [isUserLoading, setIsUserLoading] = useState(false);

  useEffect(() => {
    async function loadCurrentUser() {
      if (!token) {
        setCurrentUser(null);
        return;
      }

      setIsUserLoading(true);
      try {
        const user = await getCurrentUser(token);
        setCurrentUser(user);
      } catch (error) {
        localStorage.removeItem("papersleuth_token");
        setToken(null);
        setCurrentUser(null);
        setMessage(error.message);
      } finally {
        setIsUserLoading(false);
      }
    }

    loadCurrentUser();
  }, [token]);

  function handleLogin(accessToken) {
    localStorage.setItem("papersleuth_token", accessToken);
    setToken(accessToken);
    setMessage("Logged in successfully.");
  }

  function handleDocumentUploaded(document) {
    setRefreshDocuments((value) => value + 1);
    if (document?.document_id) {
      setSelectedDocument(document);
    }
  }

  function handleLogout() {
    localStorage.removeItem("papersleuth_token");
    setToken(null);
    setCurrentUser(null);
    setSelectedDocument(null);
    setMessage("Logged out.");
    setAuthMode("login");
  }

  function switchAuthMode(nextMode) {
    setAuthMode(nextMode);
    setMessage("");
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="nav-inner">
          <div className="brand-block">
            <span className="brand-mark">PS</span>
            <div>
              <h1>PaperSleuth</h1>
              <p>Document intelligence workspace</p>
            </div>
          </div>
          <div className="header-actions">
            {token && (
              <div className="user-chip" title={currentUser?.email || "Signed in user"}>
                {isUserLoading ? "Checking session..." : currentUser?.email}
              </div>
            )}
            {token && (
              <button className="secondary-button" type="button" onClick={handleLogout}>
                Log out
              </button>
            )}
          </div>
        </div>
      </header>

      <main>
        {!token ? (
          <section className="landing-shell">
            <div className="auth-card">
              {message && <p className="notice-message">{message}</p>}
              {authMode === "login" ? (
                <LoginForm
                  onLogin={handleLogin}
                  onSwitchToSignup={() => switchAuthMode("signup")}
                />
              ) : (
                <SignupForm
                  onSignup={setMessage}
                  onSwitchToLogin={() => switchAuthMode("login")}
                />
              )}
            </div>

            <section className="hero-panel">
              <span className="eyebrow">Azure-ready document intelligence</span>
              <h2>Extract, understand, and organize documents in one place.</h2>
              <p>
                Upload scanned PDFs, extract text with OCR, and review structured
                outputs securely.
              </p>
              <div className="feature-grid" id="features">
                <div>OCR extraction</div>
                <div>JSON outputs</div>
                <div>Document history</div>
                <div>Azure-ready storage</div>
              </div>
            </section>
          </section>
        ) : (
          <section className="dashboard-shell">
            <div className="dashboard-intro">
              <div>
                <span className="eyebrow">Dashboard</span>
                <h2>Review your document pipeline</h2>
                <p>Upload PDFs, track processing status, and inspect extracted outputs.</p>
              </div>
              <div className="session-card">
                <span className="status-dot" />
                <div>
                  <strong>{isUserLoading ? "Loading account..." : "Signed in"}</strong>
                  <p>{currentUser?.email || "Session active"}</p>
                </div>
              </div>
            </div>

            <div className="dashboard-grid">
              <section className="panel upload-panel">
                <DocumentUpload token={token} onUploaded={handleDocumentUploaded} />
              </section>
              <section className="panel history-panel">
                <DocumentList
                  token={token}
                  refreshKey={refreshDocuments}
                  selectedDocumentId={selectedDocument?.document_id}
                  onSelectDocument={setSelectedDocument}
                />
              </section>
              <section className="panel detail-panel">
                <DocumentDetail document={selectedDocument} token={token} />
              </section>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
