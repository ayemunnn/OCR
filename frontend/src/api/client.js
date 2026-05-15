const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }

  return data;
}

export function signup({ email, password, fullName }) {
  return request("/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email,
      password,
      full_name: fullName || null,
    }),
  });
}

export function login({ email, password }) {
  return request("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function getCurrentUser(token) {
  return request("/auth/me", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

export function uploadDocument({ token, file }) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/documents/process", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });
}

export function getDocuments(token) {
  return request("/documents", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

export function getDocument(documentId, token) {
  return request(`/documents/${documentId}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

export function getDocumentText(documentId, token) {
  return request(`/documents/${documentId}/text`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

export function getDocumentJson(documentId, token) {
  return request(`/documents/${documentId}/json`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}
