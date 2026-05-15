import React, { useEffect, useState } from "react";

import { getDocumentJson, getDocumentText } from "../api/client";

function DocumentDetail({ document, token }) {
  const [content, setContent] = useState("");
  const [contentType, setContentType] = useState("");
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setContent("");
    setContentType("");
    setMessage("");
  }, [document?.document_id]);

  async function handleViewText() {
    if (!document) {
      return;
    }

    setIsLoading(true);
    setMessage("");
    setContent("");
    setContentType("");

    try {
      const data = await getDocumentText(document.document_id, token);
      setContent(data.text || "No extracted text was returned.");
      setContentType("Extracted text");
    } catch (error) {
      setMessage(error.message || "No extracted text is available.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleViewJson() {
    if (!document) {
      return;
    }

    setIsLoading(true);
    setMessage("");
    setContent("");
    setContentType("");

    try {
      const data = await getDocumentJson(document.document_id, token);
      setContent(JSON.stringify(data.output, null, 2));
      setContentType("JSON output");
    } catch (error) {
      setMessage(error.message || "No JSON output is available.");
    } finally {
      setIsLoading(false);
    }
  }

  if (!document) {
    return (
      <div>
        <h2>Selected document detail panel</h2>
        <p>Select a document to view metadata, extracted text, or JSON output.</p>
      </div>
    );
  }

  return (
    <div>
      <h2>Selected document detail panel</h2>
      <p>
        <strong>{document.original_filename}</strong>
      </p>
      <p>Status: {document.status}</p>
      <p>Created: {new Date(document.created_at).toLocaleString()}</p>
      <p>Document ID: {document.document_id}</p>
      <p>
        <span className="pill">
          Text: {document.has_extracted_text ? "available" : "not available"}
        </span>{" "}
        <span className="pill">
          JSON: {document.has_output_json ? "available" : "not available"}
        </span>
      </p>

      <div className="detail-actions">
        <button
          className="secondary-button"
          type="button"
          onClick={handleViewText}
          disabled={!document.has_extracted_text || isLoading}
        >
          View extracted text
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={handleViewJson}
          disabled={!document.has_output_json || isLoading}
        >
          View JSON output
        </button>
      </div>

      {isLoading && <p>Loading document details...</p>}
      {message && <p className="error-message">{message}</p>}
      {content && (
        <>
          <h3>{contentType}</h3>
          <pre className="detail-pre">{content}</pre>
        </>
      )}
    </div>
  );
}

export default DocumentDetail;
