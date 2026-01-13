# forecaster/utils/send_email.py

import os
import smtplib
from email.message import EmailMessage
from fpdf import FPDF


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _latin1_safe(text: str) -> str:
    # Strip emojis / unsupported unicode for FPDF
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _wrap_long_tokens(text: str, max_len: int = 100) -> str:
    """
    Hard-wrap very long unbroken strings.
    """
    out = []
    for line in text.split("\n"):
        while len(line) > max_len:
            out.append(line[:max_len])
            line = line[max_len:]
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------
# PDF BUILDER (BULLETPROOF)
# ---------------------------------------------------------

def build_pdf_from_text(text: str, out_path: str):
    text = _latin1_safe(text)
    text = _wrap_long_tokens(text)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=10)

    # IMPORTANT: use write(), not multi_cell()
    for line in text.split("\n"):
        pdf.write(6, line)
        pdf.ln(6)

    pdf.output(out_path)


# ---------------------------------------------------------
# EMAIL SENDER (NON-BLOCKING)
# ---------------------------------------------------------

def send_pdf_email(
    to_email: str,
    subject: str,
    body: str,
    pdf_path: str,
):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not all([smtp_host, smtp_user, smtp_pass]):
        print("⚠️  Email skipped: SMTP credentials not set")
        return False

    try:
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with open(pdf_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(pdf_path),
            )

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"✅ PDF email sent to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ Email failed: SMTP authentication error")
        return False

    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False
