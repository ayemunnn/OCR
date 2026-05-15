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
        <div>
          <h1>PaperSleuth</h1>
          <p>OCR and document extraction workspace</p>
        </div>
        {token && (
          <button className="secondary-button" type="button" onClick={handleLogout}>
            Log out
          </button>
        )}
      </header>

      <main className="app-main">
        <aside>
          <section className="panel">
            <h2>Auth panel</h2>
            {message && <p className="message">{message}</p>}
            {!token ? (
              <div className="auth-grid">
                <SignupForm onSignup={setMessage} />
                <LoginForm onLogin={handleLogin} />
              </div>
            ) : (
              <p>
                Logged in as{" "}
                <strong>{isUserLoading ? "loading..." : currentUser?.email}</strong>
              </p>
            )}
          </section>

          {token && (
            <section className="panel">
              <DocumentUpload
                token={token}
                onUploaded={() => setRefreshDocuments((value) => value + 1)}
              />
            </section>
          )}
        </aside>

        <section className="stack">
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
            <section className="panel">
              <h2>Document workspace</h2>
              <p>Sign up or log in to upload PDFs and view document history.</p>
            </section>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
