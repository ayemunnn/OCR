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
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <span className="brand-mark">PS</span>
          <h1>PaperSleuth</h1>
          <p>OCR review and document extraction workspace</p>
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
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Account</span>
                <h2>{token ? "Session" : "Get started"}</h2>
              </div>
            </div>
            {message && <p className="notice-message">{message}</p>}
            {!token ? (
              <div className="auth-grid">
                <SignupForm onSignup={setMessage} />
                <LoginForm onLogin={handleLogin} />
              </div>
            ) : (
              <div className="session-card">
                <span className="status-dot" />
                <div>
                  <strong>{isUserLoading ? "Loading account..." : "Signed in"}</strong>
                  <p>{currentUser?.email || "Session active"}</p>
                </div>
              </div>
            )}
          </section>

          {token && (
            <section className="panel">
              <DocumentUpload
                token={token}
                onUploaded={handleDocumentUploaded}
              />
            </section>
          )}
        </aside>

        <section className="workspace-grid">
          {token ? (
            <>
              <section className="panel">
                <DocumentList
                  token={token}
                  refreshKey={refreshDocuments}
                  selectedDocumentId={selectedDocument?.document_id}
                  onSelectDocument={setSelectedDocument}
                />
              </section>
              <section className="panel">
                <DocumentDetail document={selectedDocument} token={token} />
              </section>
            </>
          ) : (
            <section className="panel empty-workspace">
              <span className="eyebrow">Workspace</span>
              <h2>Upload, extract, and review documents from one place.</h2>
              <p>Sign up or log in to start processing PDFs and building document history.</p>
            </section>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
