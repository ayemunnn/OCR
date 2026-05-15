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
      onUploaded();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>Upload document</h2>
      <input
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => setFile(event.target.files?.[0] || null)}
      />
      <button type="submit" disabled={isUploading}>
        {isUploading ? "Uploading..." : "Upload PDF"}
      </button>
      {message && <p>{message}</p>}
    </form>
  );
}

export default DocumentUpload;
