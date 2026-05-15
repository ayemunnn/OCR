import React, { useEffect, useState } from "react";

import { getDocuments } from "../api/client";

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
      <h2>Document history panel</h2>
      {isLoading && <p>Loading documents...</p>}
      {message && <p className="error-message">{message}</p>}
      {!isLoading && documents.length === 0 && <p>No documents processed yet.</p>}
      {documents.length > 0 && (
        <table className="document-table">
          <thead>
            <tr>
              <th align="left">Filename</th>
              <th align="left">Status</th>
              <th align="left">Created</th>
              <th align="left">Text</th>
              <th align="left">JSON</th>
              <th align="left">Action</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((document) => (
              <tr key={document.document_id}>
                <td>{document.original_filename}</td>
                <td>{document.status}</td>
                <td>{new Date(document.created_at).toLocaleString()}</td>
                <td>{document.has_extracted_text ? "Yes" : "No"}</td>
                <td>{document.has_output_json ? "Yes" : "No"}</td>
                <td>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onSelectDocument(document)}
                    disabled={selectedDocumentId === document.document_id}
                  >
                    {selectedDocumentId === document.document_id ? "Selected" : "Select"}
                  </button>
                  <div className="document-id">{document.document_id}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default DocumentList;
