import React, { useState } from "react";

import { uploadDocument } from "../api/client";

function DocumentUpload({ token, onUploaded }) {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setMessage("Please select a PDF file.");
      return;
    }

    setIsUploading(true);
    setMessage("");

    try {
      const result = await uploadDocument({ token, file });
      setMessage(result.message || result.status || "Document uploaded.");
      onUploaded(result);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form className="form-stack" onSubmit={handleSubmit}>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Process</span>
          <h2>Upload PDF</h2>
        </div>
      </div>
      <input
        className="file-input"
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => setFile(event.target.files?.[0] || null)}
      />
      {file && (
        <div className="file-summary">
          <strong>{file.name}</strong>
          <span>{Math.max(1, Math.round(file.size / 1024))} KB</span>
        </div>
      )}
      <button className="primary-button" type="submit" disabled={isUploading}>
        {isUploading ? "Processing..." : "Upload and process"}
      </button>
      {message && <p className="notice-message">{message}</p>}
    </form>
  );
}

export default DocumentUpload;
