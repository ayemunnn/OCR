import React, { useEffect, useState } from "react";

import { getDocuments } from "../api/client";

function DocumentList({ token, refreshKey }) {
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
      } catch (error) {
        setMessage(error.message);
      } finally {
        setIsLoading(false);
      }
    }

    loadDocuments();
  }, [token, refreshKey]);

  return (
    <div>
      <h2>Document history</h2>
      {isLoading && <p>Loading documents...</p>}
      {message && <p>{message}</p>}
      {!isLoading && documents.length === 0 && <p>No documents processed yet.</p>}
      {documents.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th align="left">Filename</th>
              <th align="left">Status</th>
              <th align="left">Created</th>
              <th align="left">Document ID</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((document) => (
              <tr key={document.document_id}>
                <td>{document.original_filename}</td>
                <td>{document.status}</td>
                <td>{new Date(document.created_at).toLocaleString()}</td>
                <td>{document.document_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default DocumentList;
