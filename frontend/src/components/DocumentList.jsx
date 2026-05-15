import React, { useEffect, useState } from "react";

import { getDocumentJson, getDocumentText, getDocuments } from "../api/client";

const detailPanelStyle = {
  background: "#f7f7f7",
  border: "1px solid #ddd",
  borderRadius: "6px",
  marginTop: "16px",
  padding: "16px",
};

const preStyle = {
  background: "#fff",
  border: "1px solid #ddd",
  borderRadius: "6px",
  maxHeight: "360px",
  overflow: "auto",
  padding: "12px",
  whiteSpace: "pre-wrap",
};

function DocumentList({ token, refreshKey }) {
  const [documents, setDocuments] = useState([]);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [detail, setDetail] = useState(null);
  const [detailMessage, setDetailMessage] = useState("");
  const [isDetailLoading, setIsDetailLoading] = useState(false);

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

  async function handleViewText(documentId) {
    setIsDetailLoading(true);
    setDetailMessage("");
    setDetail(null);

    try {
      const data = await getDocumentText(documentId, token);
      setDetail({
        title: `Extracted text: ${documentId}`,
        content: data.text || "No extracted text was returned.",
      });
    } catch (error) {
      setDetailMessage(error.message || "No extracted text is available.");
    } finally {
      setIsDetailLoading(false);
    }
  }

  async function handleViewJson(documentId) {
    setIsDetailLoading(true);
    setDetailMessage("");
    setDetail(null);

    try {
      const data = await getDocumentJson(documentId, token);
      setDetail({
        title: `JSON output: ${documentId}`,
        content: JSON.stringify(data.output, null, 2),
      });
    } catch (error) {
      setDetailMessage(error.message || "No JSON output is available.");
    } finally {
      setIsDetailLoading(false);
    }
  }

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
              <th align="left">Text</th>
              <th align="left">JSON</th>
              <th align="left">Document ID</th>
              <th align="left">Actions</th>
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
                <td>{document.document_id}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => handleViewText(document.document_id)}
                    disabled={!document.has_extracted_text || isDetailLoading}
                  >
                    View Text
                  </button>{" "}
                  <button
                    type="button"
                    onClick={() => handleViewJson(document.document_id)}
                    disabled={!document.has_output_json || isDetailLoading}
                  >
                    View JSON
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {(isDetailLoading || detailMessage || detail) && (
        <div style={detailPanelStyle}>
          {isDetailLoading && <p>Loading document details...</p>}
          {detailMessage && <p>{detailMessage}</p>}
          {detail && (
            <>
              <h3>{detail.title}</h3>
              <pre style={preStyle}>{detail.content}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default DocumentList;
