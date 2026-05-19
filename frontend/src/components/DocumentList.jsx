import React, { useEffect, useState } from "react";

import { getDocuments } from "../api/client";

function formatDate(value) {
  if (!value) {
    return "Unknown";
  }
  return new Date(value).toLocaleString();
}

function DocumentList({ token, refreshKey, selectedDocumentId, onSelectDocument }) {
  const [documents, setDocuments] = useState([]);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadDocuments() {
      setIsLoading(true);
      setMessage("");

      try {
        const data = await getDocuments(token);
        setDocuments(data.documents || []);
        if (data.documents?.length && !selectedDocumentId) {
          onSelectDocument(data.documents[0]);
        }
      } catch (error) {
        setMessage(error.message);
      } finally {
        setIsLoading(false);
      }
    }

    loadDocuments();
  }, [token, refreshKey, selectedDocumentId, onSelectDocument]);

  return (
    <div>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Library</span>
          <h2>Document history</h2>
        </div>
        <span className="count-badge">{documents.length}</span>
      </div>
      {isLoading && <p>Loading documents...</p>}
      {message && <p className="error-message">{message}</p>}
      {!isLoading && documents.length === 0 && (
        <div className="empty-state">
          <h3>No documents yet</h3>
          <p>Upload your first PDF to see OCR text, JSON output, and processing status here.</p>
        </div>
      )}
      {documents.length > 0 && (
        <div className="document-list">
          {documents.map((document) => {
            const isSelected = selectedDocumentId === document.document_id;
            return (
              <button
                className={`document-row ${isSelected ? "is-selected" : ""}`}
                key={document.document_id}
                type="button"
                onClick={() => onSelectDocument(document)}
              >
                <span className="document-main">
                  <strong>{document.original_filename}</strong>
                  <span>{formatDate(document.created_at)}</span>
                  <span className="document-id">{document.document_id}</span>
                </span>
                <span className="document-meta">
                  <span className="status-pill">{document.status}</span>
                  <span className={document.has_extracted_text ? "pill good" : "pill muted"}>
                    Text
                  </span>
                  <span className={document.has_output_json ? "pill good" : "pill muted"}>
                    JSON
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default DocumentList;
