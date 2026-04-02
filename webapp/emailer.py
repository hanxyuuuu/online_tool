from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Tuple

from .schemas import PredictionResponse


@dataclass
class EmailSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    sender: str
    use_starttls: bool
    use_ssl: bool
    subject_prefix: str


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_email_settings() -> EmailSettings | None:
    host = os.getenv("TFDNA_SMTP_HOST", "").strip()
    if not host:
        return None

    username = os.getenv("TFDNA_SMTP_USERNAME", "").strip() or None
    password = os.getenv("TFDNA_SMTP_PASSWORD", "").strip() or None
    sender = os.getenv("TFDNA_SMTP_FROM", "").strip() or username
    if not sender:
        return None

    return EmailSettings(
        host=host,
        port=int(os.getenv("TFDNA_SMTP_PORT", "587")),
        username=username,
        password=password,
        sender=sender,
        use_starttls=_bool_env("TFDNA_SMTP_STARTTLS", True),
        use_ssl=_bool_env("TFDNA_SMTP_SSL", False),
        subject_prefix=os.getenv("TFDNA_EMAIL_SUBJECT_PREFIX", "[TF-DNA Binding Atlas]"),
    )


def validate_email_address(email: str | None) -> str | None:
    if not email:
        return None
    candidate = email.strip()
    _, parsed = parseaddr(candidate)
    if not parsed or "@" not in parsed or "." not in parsed.split("@", 1)[-1]:
        raise ValueError("Please enter a valid email address.")
    return parsed


def _top_positions(sequence: str, scores: list[float], count: int = 3) -> list[Tuple[int, str, float]]:
    pairs = sorted(
        enumerate(scores, start=1),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:count]
    return [(index, sequence[index - 1], score) for index, score in pairs]


def _build_email_message(recipient: str, result: PredictionResponse, settings: EmailSettings) -> EmailMessage:
    dna_top = _top_positions(result.normalized_dna_sequence, result.dna_importance_raw)
    protein_top = _top_positions(result.normalized_protein_sequence, result.protein_importance_raw)

    text_lines = [
        "TF-DNA Binding Atlas prediction result",
        "",
        f"Predicted class: {result.predicted_class_text} ({result.predicted_label})",
        f"Probability: {result.probability:.6f}",
        f"Raw logit: {result.logit:.6f}",
        "",
        "Input summary:",
        f"- DNA length: {result.input_summary.original_dna_length} -> {result.input_summary.normalized_dna_length}",
        f"- Protein length: {result.input_summary.original_protein_length} -> {result.input_summary.normalized_protein_length}",
    ]

    if result.input_summary.messages:
        text_lines.append("- Notes:")
        text_lines.extend(f"  * {message}" for message in result.input_summary.messages)

    text_lines.extend(
        [
            "",
            "Top DNA positions:",
            *[
                f"- {base}{position}: {score:.6f}"
                for position, base, score in dna_top
            ],
            "",
            "Top protein positions:",
            *[
                f"- {residue}{position}: {score:.6f}"
                for position, residue, score in protein_top
            ],
            "",
            "Normalized DNA sequence:",
            result.normalized_dna_sequence,
            "",
            "Normalized protein sequence:",
            result.normalized_protein_sequence,
        ]
    )

    dna_rows = "".join(
        f"<li><strong>{base}{position}</strong>: {score:.6f}</li>"
        for position, base, score in dna_top
    )
    protein_rows = "".join(
        f"<li><strong>{residue}{position}</strong>: {score:.6f}</li>"
        for position, residue, score in protein_top
    )
    notes_html = "".join(f"<li>{message}</li>" for message in result.input_summary.messages)

    message = EmailMessage()
    message["Subject"] = f"{settings.subject_prefix} Prediction result"
    message["From"] = settings.sender
    message["To"] = recipient
    message.set_content("\n".join(text_lines))
    message.add_alternative(
        f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #18222a;">
            <h2>TF-DNA Binding Atlas prediction result</h2>
            <p><strong>Predicted class:</strong> {result.predicted_class_text} ({result.predicted_label})</p>
            <p><strong>Probability:</strong> {result.probability:.6f}<br>
               <strong>Raw logit:</strong> {result.logit:.6f}</p>
            <h3>Input summary</h3>
            <ul>
              <li>DNA length: {result.input_summary.original_dna_length} -> {result.input_summary.normalized_dna_length}</li>
              <li>Protein length: {result.input_summary.original_protein_length} -> {result.input_summary.normalized_protein_length}</li>
              {notes_html}
            </ul>
            <h3>Top DNA positions</h3>
            <ul>{dna_rows}</ul>
            <h3>Top protein positions</h3>
            <ul>{protein_rows}</ul>
            <h3>Normalized DNA sequence</h3>
            <p style="font-family: Consolas, monospace; word-break: break-all;">{result.normalized_dna_sequence}</p>
            <h3>Normalized protein sequence</h3>
            <p style="font-family: Consolas, monospace; word-break: break-all;">{result.normalized_protein_sequence}</p>
          </body>
        </html>
        """,
        subtype="html",
    )
    return message


def send_prediction_email(recipient: str, result: PredictionResponse) -> tuple[str, str]:
    settings = load_email_settings()
    if settings is None:
        return (
            "not_configured",
            "Prediction finished, but email delivery is not configured on this server.",
        )

    message = _build_email_message(recipient, result, settings)
    smtp_cls = smtplib.SMTP_SSL if settings.use_ssl else smtplib.SMTP
    with smtp_cls(settings.host, settings.port, timeout=30) as smtp:
        smtp.ehlo()
        if settings.use_starttls and not settings.use_ssl:
            smtp.starttls()
            smtp.ehlo()
        if settings.username and settings.password:
            smtp.login(settings.username, settings.password)
        smtp.send_message(message)

    return ("sent", f"Prediction result was emailed to {recipient}.")
