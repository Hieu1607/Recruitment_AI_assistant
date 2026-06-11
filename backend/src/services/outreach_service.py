from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from typing import Any
import re

from src.models.candidate_profile import CandidateProfile
from src.models.job import Job


PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")
ALLOWED_TAGS = {"p", "br", "strong", "b", "em", "i", "u", "ul", "ol", "li", "a", "span", "img"}
ALLOWED_ATTRS = {
    "a": {"href", "target", "rel"},
    "span": {"style"},
    "img": {"src", "alt", "title", "width", "height"},
}
SAFE_STYLE_PARTS = {"color", "background-color", "text-decoration"}


def build_render_variables(candidate: CandidateProfile, job: Job | None, company_name: str | None = None) -> dict[str, str]:
    candidate_name = candidate.full_name or "candidate"
    return {
        "candidate_name": candidate_name,
        "candidate_email": candidate.email or "",
        "job_title": job.title if job is not None else "",
        "company_name": company_name or "",
    }


def render_template_string(template: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, "")

    return PLACEHOLDER_PATTERN.sub(replace, template)


class _HtmlSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    @staticmethod
    def _sanitize_style(value: str) -> str:
        safe_parts: list[str] = []
        for chunk in value.split(";"):
            if ":" not in chunk:
                continue
            prop, raw = chunk.split(":", 1)
            normalized_prop = prop.strip().lower()
            if normalized_prop not in SAFE_STYLE_PARTS:
                continue
            safe_parts.append(f"{normalized_prop}: {raw.strip()}")
        return "; ".join(safe_parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in ALLOWED_TAGS:
            return
        rendered_attrs: list[str] = []
        allowed = ALLOWED_ATTRS.get(tag, set())
        for name, value in attrs:
            if name not in allowed or value is None:
                continue
            if name in {"href", "src"} and value.lower().startswith("javascript:"):
                continue
            if name == "style":
                value = self._sanitize_style(value)
                if not value:
                    continue
            rendered_attrs.append(f'{name}="{escape(value, quote=True)}"')
        attr_text = f" {' '.join(rendered_attrs)}" if rendered_attrs else ""
        self._parts.append(f"<{tag}{attr_text}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in ALLOWED_TAGS and tag != "br" and tag != "img":
            self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._parts.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._parts)


def sanitize_email_html(value: str) -> str:
    parser = _HtmlSanitizer()
    parser.feed(value or "")
    parser.close()
    return parser.get_html().strip()


def html_to_plain_text(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\s*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_rich_message(*, body_text: str | None, body_html: str | None) -> tuple[str, str]:
    sanitized_html = sanitize_email_html(body_html or "")
    normalized_text = (body_text or "").strip()
    if not normalized_text and sanitized_html:
        normalized_text = html_to_plain_text(sanitized_html)
    if not sanitized_html and normalized_text:
        paragraphs = "</p><p>".join(escape(part) for part in normalized_text.split("\n\n"))
        sanitized_html = f"<p>{paragraphs}</p>" if paragraphs else ""
        sanitized_html = sanitized_html.replace("\n", "<br>")
    return normalized_text.strip(), sanitized_html.strip()
