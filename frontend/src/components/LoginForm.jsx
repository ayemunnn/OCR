import React, { useState } from "react";

import { login } from "../api/client";

function LoginForm({ onLogin, onSwitchToSignup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setMessage("");

    try {
      const data = await login({ email, password });
      onLogin(data.access_token);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form className="form-stack" onSubmit={handleSubmit}>
      <div className="auth-copy">
        <h2>Welcome back</h2>
        <p>Log in to continue processing and reviewing your documents.</p>
      </div>
      <label>
        Email
        <input
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          required
        />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          required
        />
      </label>
      <button className="primary-button" type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Logging in..." : "Log in"}
      </button>
      {message && <p className="error-message">{message}</p>}
      <p className="auth-switch">
        New to PaperSleuth?{" "}
        <button type="button" onClick={onSwitchToSignup}>
          Create an account
        </button>
      </p>
    </form>
  );
}

export default LoginForm;
