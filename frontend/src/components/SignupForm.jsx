import React, { useState } from "react";

import { signup } from "../api/client";

function SignupForm({ onSignup, onSwitchToLogin }) {
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setMessage("");

    try {
      await signup({ email, password, fullName });
      setMessage("Signup successful. You can log in now.");
      onSignup("Signup successful. You can log in now.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form className="form-stack" onSubmit={handleSubmit}>
      <div className="auth-copy">
        <h2>Create your account</h2>
        <p>Start extracting and organizing scanned PDFs with PaperSleuth.</p>
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
        Full name
        <input
          type="text"
          value={fullName}
          onChange={(event) => setFullName(event.target.value)}
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
        {isSubmitting ? "Creating..." : "Create account"}
      </button>
      {message && <p className="message">{message}</p>}
      <p className="auth-switch">
        Already have an account?{" "}
        <button type="button" onClick={onSwitchToLogin}>
          Log in
        </button>
      </p>
    </form>
  );
}

export default SignupForm;
