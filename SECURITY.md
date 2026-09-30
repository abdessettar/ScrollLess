# Security

## Reporting a vulnerability

Please report vulnerabilities privately with the **Report a vulnerability** button in the
repository's **Security** tab, rather than opening a public issue.

## Handling credentials

- API keys and SMTP credentials are read from environment variables, loaded from `.env`.
  `.env` and your personal `config.*.yaml` files are git-ignored. Only `.env.example` and
  `examples/` are tracked.
- Use an app-specific SMTP password (such as a Gmail App Password), never your main account password.
- Reddit OAuth, when enabled, uses app-only read access and never your account password.
- Post titles, bodies and comments are sent to the configured LLM provider for summarization.
  Only public content is collected.
- If a key is ever committed, revoke it at the provider right away. Removing it from git history is not enough.
