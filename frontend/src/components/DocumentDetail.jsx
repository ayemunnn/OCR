import React, { useEffect, useState } from "react";

import { getDocumentJson, getDocumentText } from "../api/client";

function formatDate(value) {
  if (!value) {
    return "Unknown";
  }
  return new Date(value).toLocaleString();
}

function buildJsonFilename(document) {
  const baseName = document.original_filename
    .replace(/\.pdf$/i, "")
    .replace(/[^a-z0-9-_]+/gi, "_")
    .replace(/^_+|_+$/g, "");
  return `${baseName || document.document_id || "papersleuth_output"}.json`;
}

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

  async function handleDownloadJson() {
    if (!document) {
      return;
    }

    setIsLoading(true);
    setMessage("");

    try {
      const data = await getDocumentJson(document.document_id, token);
      const jsonText = JSON.stringify(data.output, null, 2);
      const blob = new Blob([jsonText], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = window.document.createElement("a");
      link.href = url;
      link.download = buildJsonFilename(document);
      window.document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setMessage("JSON file downloaded.");
    } catch (error) {
      setMessage(error.message || "No JSON output is available to download.");
    } finally {
      setIsLoading(false);
    }
  }

  if (!document) {
    return (
      <div className="empty-state detail-empty">
        <span className="eyebrow">Details</span>
        <h2>Select a document</h2>
        <p>Choose a processed PDF from the history panel to view metadata, extracted text, or JSON output.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Details</span>
          <h2>{document.original_filename}</h2>
        </div>
      </div>

      <dl className="metadata-grid">
        <div>
          <dt>Status</dt>
          <dd>{document.status}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{formatDate(document.created_at)}</dd>
        </div>
        <div>
          <dt>Document ID</dt>
          <dd className="document-id">{document.document_id}</dd>
        </div>
        <div>
          <dt>Outputs</dt>
          <dd>
            <span className={document.has_extracted_text ? "pill good" : "pill muted"}>
              Text {document.has_extracted_text ? "available" : "missing"}
            </span>
            <span className={document.has_output_json ? "pill good" : "pill muted"}>
              JSON {document.has_output_json ? "available" : "missing"}
            </span>
          </dd>
        </div>
      </dl>

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
        <button
          className="secondary-button"
          type="button"
          onClick={handleDownloadJson}
          disabled={!document.has_output_json || isLoading}
        >
          Download JSON
        </button>
      </div>

      {isLoading && <p>Loading document details...</p>}
      {message && <p className="error-message">{message}</p>}
      {content && (
        <div className="output-panel">
          <div className="output-heading">
            <h3>{contentType}</h3>
          </div>
          <pre className="detail-pre">{content}</pre>
        </div>
      )}
    </div>
  );
}

export default DocumentDetail;
