"""Standalone rendered-HTML feature extraction for Phase 3.

This module deliberately does not load a phishing model or perform prediction.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse


_LOG = logging.getLogger(__name__)


def _diagnostic_log(message: str, *args: object) -> None:
    """Log diagnostics without requiring a usable Windows console handle."""
    try:
        _LOG.info(message, *args)
    except (OSError, ValueError):
        pass

try:
    import pandas as pd
except ImportError:
    pd = Any  # type: ignore[misc,assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = Any  # type: ignore[misc,assignment]

try:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright, TimeoutError as PlaywrightTimeoutError, sync_playwright
except ImportError:  # Keep the module importable until Phase 3 dependencies are installed.
    Browser = BrowserContext = Page = Playwright = Any  # type: ignore[misc,assignment]
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None


class HTMLFeatureExtractionError(RuntimeError):
    """Raised when rendered HTML cannot be retrieved safely."""


class AntiBotProtectionError(HTMLFeatureExtractionError):
    """Raised when an anti-bot challenge page is detected instead of the actual content."""

    def __init__(self, message: str, markers: list[str] | None = None) -> None:
        super().__init__(message)
        self.markers = list(markers or [])


# Generic interstitial / bot-protection markers (never website-specific).
BOT_PROTECTION_MARKERS: tuple[str, ...] = (
    "just a moment",
    "cf-browser-verification",
    "challenge-platform",
    "attention required",
    "security check",
    "robot check",
    "checking your browser",
    "awswaf",
    "awswafcookiedomainlist",
    "token.awswaf.com",
    "challenge-container",
    "verify that you're not a robot",
    "aws waf",
    "incapsula",
    "sucuri_cloudproxy",
    "akamai bot manager",
)


def detect_bot_protection_markers(title_text: str, content_text: str) -> list[str]:
    """Return matching generic challenge/WAF markers found in title or HTML."""
    haystack = f"{title_text or ''}\n{content_text or ''}".lower()
    found: list[str] = []
    for marker in BOT_PROTECTION_MARKERS:
        if marker.lower() in haystack:
            found.append(marker)
    return found


def parse_host_identity(url_or_host: str) -> dict[str, str]:
    """Parse a URL or hostname with the public-suffix list (tldextract).

    brand_name is the registrable *domain* label (e.g. kongu for kongu.ac.in),
    never a public-suffix component such as ``ac`` or ``gov``.
    """
    empty = {
        "host": "",
        "subdomain": "",
        "domain": "",
        "suffix": "",
        "registered_domain": "",
        "brand_name": "",
    }
    raw = (url_or_host or "").strip()
    if not raw:
        return dict(empty)

    host = ""
    try:
        candidate = raw if "://" in raw else f"https://{raw}"
        host = (urlparse(candidate).hostname or "").lower()
    except Exception:
        host = raw.split("/")[0].lower()

    try:
        import tldextract

        ext = tldextract.extract(raw)
        domain = (ext.domain or "").lower()
        suffix = (ext.suffix or "").lower()
        subdomain = (ext.subdomain or "").lower()
        registered = f"{domain}.{suffix}" if domain and suffix else domain
        brand = _safe_brand_label(host, domain, suffix)
        if brand and brand != domain:
            registered = f"{brand}.{suffix}" if brand and suffix else brand
            domain = brand
        return {
            "host": host,
            "subdomain": subdomain,
            "domain": domain,
            "suffix": suffix,
            "registered_domain": registered,
            "brand_name": brand,
        }
    except Exception:
        host_no_www = host[4:] if host.startswith("www.") else host
        parts = [p for p in host_no_www.split(".") if p]
        suffix_guess = ".".join(parts[-2:]) if len(parts) >= 3 else (parts[-1] if parts else "")
        domain_guess = parts[-2] if len(parts) >= 2 else (parts[0] if parts else host)
        brand = _safe_brand_label(host, domain_guess, suffix_guess)
        registered = f"{brand}.{suffix_guess}" if brand and suffix_guess else (brand or host_no_www)
        return {
            "host": host,
            "subdomain": "",
            "domain": brand,
            "suffix": suffix_guess,
            "registered_domain": registered,
            "brand_name": brand,
        }


class CDNErrorPageError(HTMLFeatureExtractionError):
    """Raised when a CDN error page is detected instead of the actual content."""


# Labels that are public-suffix components, never a site brand (e.g. "ac" in ac.in).
_SUFFIX_LIKE_LABELS = frozenset({
    "ac", "co", "com", "net", "org", "gov", "edu", "res", "gen", "firm",
    "ind", "info", "biz", "web", "int", "mil", "nic", "or", "ne", "go",
    "in", "uk", "au", "us", "za", "jp", "br", "cn", "io", "ai",
})


def _safe_brand_label(host: str, domain: str, suffix: str) -> str:
    """Return the registrable brand label, never a public-suffix token such as ac/gov."""
    suffix_labels = [p for p in (suffix or "").lower().split(".") if p]
    host_labels = [p for p in (host or "").lower().split(".") if p and p != "www"]
    remainder: list[str]
    if suffix_labels and len(host_labels) >= len(suffix_labels) and host_labels[-len(suffix_labels):] == suffix_labels:
        remainder = host_labels[:-len(suffix_labels)]
    else:
        remainder = [p for p in host_labels if p not in suffix_labels]
    remainder = [p for p in remainder if p not in _SUFFIX_LIKE_LABELS]
    if remainder:
        return remainder[-1]
    domain_l = (domain or "").lower()
    if domain_l and domain_l not in _SUFFIX_LIKE_LABELS and domain_l not in suffix_labels:
        return domain_l
    return remainder[0] if remainder else domain_l


def wait_for_rendered_content(page: Page, timeout_ms: int = 12_000) -> dict[str, int]:
    """Poll until the document looks like a real page, not a splash or WAF stub.

    Generic heuristics only: enough anchors, visible text, or a large DOM.
    """
    stats = {"anchors": 0, "images": 0, "text_len": 0, "elements": 0, "html_len": 0}
    if page is None:
        return stats
    deadline = time.perf_counter() + max(timeout_ms, 0) / 1000.0
    try:
        page.wait_for_selector("a[href]", timeout=min(timeout_ms, 10_000))
    except Exception:
        pass
    js = """() => {
        const body = document.body;
        const text = (body && body.innerText || '').trim();
        const html = document.documentElement ? document.documentElement.innerHTML : '';
        return {
            anchors: document.querySelectorAll('a[href]').length,
            images: document.querySelectorAll('img').length,
            text_len: text.length,
            elements: document.getElementsByTagName('*').length,
            html_len: html.length
        };
    }"""
    while True:
        try:
            raw = page.evaluate(js) or {}
            stats = {
                "anchors": int(raw.get("anchors") or 0),
                "images": int(raw.get("images") or 0),
                "text_len": int(raw.get("text_len") or 0),
                "elements": int(raw.get("elements") or 0),
                "html_len": int(raw.get("html_len") or 0),
            }
        except Exception:
            pass
        if stats["anchors"] >= 8 or stats["text_len"] >= 800 or stats["elements"] >= 180:
            return stats
        if time.perf_counter() >= deadline:
            return stats
        try:
            page.wait_for_timeout(400)
        except Exception:
            return stats



class HTMLFeatureExtractor:
    """Render a web page with Chromium and extract basic HTML-level signals.

    Create one instance per workflow, call :meth:`launch_browser`, retrieve one
    or more pages, and always call :meth:`close_browser` when finished.
    """

    def __init__(self, timeout_ms: int = 30_000) -> None:
        self.timeout_ms = timeout_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def launch_browser(self) -> None:
        """Launch headless Chromium with JavaScript enabled and a default timeout."""
        if self._browser is not None:
            return
        if sync_playwright is None:
            raise HTMLFeatureExtractionError(
                "Phase 3 dependencies are unavailable. Install them with "
                "'pip install playwright beautifulsoup4 lxml' and then run "
                "'playwright install chromium'."
            )
        try:
            self._playwright = sync_playwright().start()
            try:
                self._browser = self._playwright.chromium.launch(headless=True)
            except Exception as launch_error:
                # Fall back to a locally installed Chrome/Edge when Playwright's
                # bundled Chromium is missing (common in restricted environments).
                try:
                    self._browser = self._playwright.chromium.launch(headless=True, channel="chrome")
                    _diagnostic_log("Browser launched via system Chrome channel")
                except Exception:
                    try:
                        self._browser = self._playwright.chromium.launch(headless=True, channel="msedge")
                        _diagnostic_log("Browser launched via system Edge channel")
                    except Exception:
                        raise launch_error
            self._context = self._browser.new_context(
                java_script_enabled=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
                color_scheme="light",
            )
            self._context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                window.navigator.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)
            _diagnostic_log("Context created with stealth settings")
        except Exception as error:
            self.close_browser()
            raise HTMLFeatureExtractionError(f"Could not launch headless Chromium: {error}") from error

    def fetch_rendered_html(self, url: str) -> str:
        """Return rendered HTML after navigation, redirects, and dynamic-content wait.

        Redirects are followed by Playwright automatically. Invalid URLs,
        navigation failures, and timeouts raise :class:`HTMLFeatureExtractionError`.
        """
        parsed = urlparse(url.strip()) if isinstance(url, str) else None
        if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTMLFeatureExtractionError("URL must be a non-empty absolute HTTP or HTTPS URL.")
        if self._context is None:
            raise HTMLFeatureExtractionError("Browser is not running. Call launch_browser() first.")

        page: Page | None = None
        try:
            page = self._context.new_page()
            _diagnostic_log("Page created")
            page.set_default_timeout(self.timeout_ms)
            nav_response = None
            # Retry loop for navigation
            for attempt in range(2):
                try:
                    try:
                        _diagnostic_log("URL navigation started (networkidle) - attempt %d", attempt + 1)
                        nav_response = page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
                        _diagnostic_log("Navigation completed")
                    except PlaywrightTimeoutError:
                        # Challenge pages often never reach networkidle. Inspect the
                        # already-loaded document before a second navigation.
                        try:
                            pending_content = page.content()
                            pending_title = page.title()
                        except Exception:
                            pending_content, pending_title = "", ""
                        pending_markers = detect_bot_protection_markers(pending_title, pending_content)
                        if pending_markers:
                            HTMLFeatureExtractor._last_bot_markers = pending_markers
                            HTMLFeatureExtractor._last_html_length = len(pending_content)
                            raise AntiBotProtectionError(
                                "Anti-bot challenge detected. Cannot analyze.",
                                markers=pending_markers,
                            )
                        _diagnostic_log("URL navigation started (fallback)")
                        nav_response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                        page.wait_for_timeout(750)
                        _diagnostic_log("Navigation completed (fallback)")
                    break
                except AntiBotProtectionError:
                    raise
                except PlaywrightTimeoutError as error:
                    if attempt == 0:
                        _diagnostic_log("PlaywrightTimeoutError on attempt 1, retrying in 5 seconds")
                        page.wait_for_timeout(5000)
                    else:
                        raise error
                except Exception as error:
                    if attempt == 0:
                        _diagnostic_log("Navigation error (%s) on attempt 1, retrying in 5 seconds", error)
                        page.wait_for_timeout(5000)
                    else:
                        raise error

            def is_cdn_error(title_text: str, content_text: str) -> bool:
                t_lower = title_text.lower()
                c_lower = content_text.lower()
                cdn_signatures = [
                    "error: the request could not be satisfied",
                    "generated by cloudfront",
                    "request blocked",
                    "reference #",
                    "access denied",
                    "403 forbidden",
                    "403 error",
                    "503 service unavailable",
                    "edge error",
                    "cdn error",
                    "imperva"
                ]
                for sig in cdn_signatures:
                    if sig in t_lower or sig in c_lower:
                        return True
                return False

            content = page.content()
            title = page.title()
            markers = detect_bot_protection_markers(title, content)

            if markers:
                _diagnostic_log("Anti-bot detected (%s), waiting for challenge resolution", markers)
                page.wait_for_timeout(8000)
                content = page.content()
                title = page.title()
                markers = detect_bot_protection_markers(title, content)
                if markers:
                    HTMLFeatureExtractor._last_bot_markers = markers
                    HTMLFeatureExtractor._last_html_length = len(content)
                    HTMLFeatureExtractor._last_html_excerpt = content[:8000]
                    HTMLFeatureExtractor._last_http_status = getattr(nav_response, "status", None)
                    raise AntiBotProtectionError(
                        "Anti-bot challenge detected. Cannot analyze.",
                        markers=markers,
                    )

            readiness = wait_for_rendered_content(page, timeout_ms=min(12_000, max(self.timeout_ms, 1_000)))
            _diagnostic_log("Rendered-content wait: %s", readiness)
            content = page.content()
            title = page.title()
            markers = detect_bot_protection_markers(title, content)
            if markers:
                HTMLFeatureExtractor._last_bot_markers = markers
                HTMLFeatureExtractor._last_html_length = len(content)
                HTMLFeatureExtractor._last_html_excerpt = content[:8000]
                HTMLFeatureExtractor._last_http_status = getattr(nav_response, "status", None)
                raise AntiBotProtectionError(
                    "Anti-bot challenge detected. Cannot analyze.",
                    markers=markers,
                )

            if is_cdn_error(title, content):
                _diagnostic_log("CDN Error Page detected")
                raise CDNErrorPageError("CDN/CloudFront error page detected. Actual webpage could not be analysed.")

            _diagnostic_log("Page info: url=%s title=%s", page.url, title)
            HTMLFeatureExtractor._last_page_url = page.url
            HTMLFeatureExtractor._last_page_title = page.title()
            HTMLFeatureExtractor._last_html_length = len(content)
            HTMLFeatureExtractor._last_html_excerpt = content[:8000]
            HTMLFeatureExtractor._last_bot_markers = []
            HTMLFeatureExtractor._last_http_status = getattr(nav_response, "status", None)
            HTMLFeatureExtractor._last_readiness = readiness
            return content
        except PlaywrightTimeoutError as error:
            _LOG.exception("Timed out while retrieving rendered HTML for %r", url)
            raise HTMLFeatureExtractionError(f"Timed out after {self.timeout_ms} ms while loading {url!r}.") from error
        except (AntiBotProtectionError, CDNErrorPageError) as error:
            raise error
        except Exception as error:
            _LOG.exception("Could not retrieve rendered HTML for %r", url)
            raise HTMLFeatureExtractionError(f"Could not retrieve rendered HTML for {url!r}: {error}") from error
        finally:
            if page is not None:
                page.close()

    @staticmethod
    def parse_html(html: str) -> BeautifulSoup:
        """Parse rendered HTML using BeautifulSoup backed by lxml (or html.parser fallback)."""
        if not isinstance(html, str):
            raise ValueError("HTML must be supplied as a string.")
        if BeautifulSoup is Any:
            raise HTMLFeatureExtractionError("BeautifulSoup is not installed; install Phase 3 dependencies.")
        try:
            return BeautifulSoup(html, "lxml")
        except Exception:
            _LOG.exception("lxml parsing failed; falling back to html.parser")
            return BeautifulSoup(html, "html.parser")

    @staticmethod
    def extract_basic_information(soup: BeautifulSoup) -> dict[str, str | int]:
        """Extract the required page-level HTML information into a dictionary."""
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        description = soup.find("meta", attrs={"name": lambda value: isinstance(value, str) and value.lower() == "description"})
        meta_description = description.get("content", "").strip() if description else ""
        return {
            "page_title": title,
            "meta_description": meta_description,
            "number_of_forms": len(soup.find_all("form")),
            "number_of_images": len(soup.find_all("img")),
            "number_of_javascript_files": len(soup.find_all("script", src=True)),
            "number_of_css_files": len(soup.find_all("link", rel=lambda value: value and "stylesheet" in value)),
            "number_of_hyperlinks": len(soup.find_all("a", href=True)),
        }

    @staticmethod
    def extract_form_features(soup: BeautifulSoup, page_url: str = "") -> dict[str, int]:
        """Extract form security and credential harvesting signals from rendered HTML.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content of the page.
        page_url : str, optional
            The URL of the rendered page, used to evaluate relative vs. external form actions.

        Returns
        -------
        dict[str, int]
            Dictionary of form-level security features.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        forms = soup.find_all("form")
        num_forms = len(forms)

        # Extract registered domain of the host page for external submission checking
        identity = parse_host_identity(page_url) if page_url and isinstance(page_url, str) else parse_host_identity("")
        page_host = identity["host"]
        page_domain = identity["registered_domain"]

        num_password_inputs = 0
        num_hidden_inputs = 0
        num_text_inputs = 0
        num_submit_inputs = 0

        has_external_form_action = 0
        has_empty_or_blank_action = 0
        has_relative_form_action = 0
        has_external_action_password_form = 0

        # Count inputs globally across DOM
        for inp in soup.find_all("input"):
            itype = (inp.get("type") or "").strip().lower()
            if itype == "password":
                num_password_inputs += 1
            elif itype == "hidden":
                num_hidden_inputs += 1
            elif itype in ("", "text", "email", "number", "tel"):
                num_text_inputs += 1
            elif itype == "submit":
                num_submit_inputs += 1

        # Count buttons that act as submit buttons
        for button in soup.find_all("button"):
            btype = (button.get("type") or "").strip().lower()
            if btype in ("", "submit"):
                num_submit_inputs += 1

        for form in forms:
            action = (form.get("action") or "").strip()
            action_lower = action.lower()

            is_external = False

            if not action or action == "#" or action_lower.startswith("about:blank") or action_lower.startswith("javascript:"):
                has_empty_or_blank_action = 1
            else:
                parsed_action = urlparse(action)
                action_host = (parsed_action.hostname or "").lower()

                if not parsed_action.scheme and not action_host:
                    has_relative_form_action = 1
                elif action_host:
                    action_domain = parse_host_identity(action_host)["registered_domain"] or action_host

                    if page_domain and action_domain != page_domain:
                        is_external = True
                        has_external_form_action = 1
                    elif not page_domain and page_host and action_host != page_host:
                        is_external = True
                        has_external_form_action = 1

            form_has_password = any(
                (inp.get("type") or "").strip().lower() == "password"
                for inp in form.find_all("input")
            )
            if form_has_password and is_external:
                has_external_action_password_form = 1

        return {
            "num_forms": num_forms,
            "num_password_inputs": num_password_inputs,
            "has_external_form_action": has_external_form_action,
            "has_empty_or_blank_action": has_empty_or_blank_action,
            "has_relative_form_action": has_relative_form_action,
            "num_hidden_inputs": num_hidden_inputs,
            "num_text_inputs": num_text_inputs,
            "num_submit_inputs": num_submit_inputs,
            "has_external_action_password_form": has_external_action_password_form,
        }

    @staticmethod
    def extract_link_features(soup: BeautifulSoup, page_url: str = "") -> dict[str, float | int]:
        """Extract hyperlink, domain ratio, and anchor text mismatch signals.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content of the page.
        page_url : str, optional
            The URL of the rendered page, used to evaluate relative vs. external anchor links.

        Returns
        -------
        dict[str, float | int]
            Dictionary of link-level security features and ratio metrics.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        anchors = soup.find_all("a")
        num_links = len(anchors)

        if num_links == 0:
            return {
                "num_links": 0,
                "num_external_links": 0,
                "num_internal_links": 0,
                "num_null_self_links": 0,
                "ratio_external_links": 0.0,
                "ratio_internal_links": 0.0,
                "ratio_null_self_links": 0.0,
                "num_suspicious_anchor_text": 0,
                "has_mismatch_link_text": 0,
            }

        identity = parse_host_identity(page_url) if page_url and isinstance(page_url, str) else parse_host_identity("")
        page_host = identity["host"]
        page_domain = identity["registered_domain"]

        num_external = 0
        num_internal = 0
        num_null_self = 0
        num_suspicious_anchor_text = 0
        has_mismatch_link_text = 0

        suspicious_keywords = {
            "click here", "login", "log in", "verify", "update", "confirm",
            "account", "security", "sign in", "signin", "password", "bank",
        }

        url_regex = re.compile(r"^(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", re.IGNORECASE)

        for anchor in anchors:
            href = (anchor.get("href") or "").strip()
            href_lower = href.lower()

            is_null_self = False
            is_external = False
            is_internal = False

            if not href or href == "#" or href_lower.startswith("javascript:") or href_lower.startswith("about:blank"):
                is_null_self = True
                num_null_self += 1
            else:
                try:
                    parsed_href = urlparse(href)
                    href_host = (parsed_href.hostname or "").lower()

                    if not parsed_href.scheme and not href_host:
                        is_internal = True
                        num_internal += 1
                    elif href_host:
                        href_domain = parse_host_identity(href_host)["registered_domain"] or href_host

                        if page_domain and href_domain == page_domain:
                            is_internal = True
                            num_internal += 1
                        elif not page_domain and page_host and href_host == page_host:
                            is_internal = True
                            num_internal += 1
                        else:
                            is_external = True
                            num_external += 1
                    else:
                        is_null_self = True
                        num_null_self += 1
                except Exception:
                    is_null_self = True
                    num_null_self += 1

            anchor_text = anchor.get_text(" ", strip=True)
            anchor_text_lower = anchor_text.lower()

            if (is_external or is_null_self) and any(keyword in anchor_text_lower for keyword in suspicious_keywords):
                num_suspicious_anchor_text += 1

            match = url_regex.search(anchor_text)
            if match:
                text_domain = parse_host_identity(match.group(1).lower())["registered_domain"]

                if is_null_self or is_external:
                    has_mismatch_link_text = 1
                elif is_internal and href:
                    try:
                        parsed_href = urlparse(href)
                        actual_host = (parsed_href.hostname or page_host).lower()
                        actual_domain = parse_host_identity(actual_host)["registered_domain"] or actual_host
                        if text_domain != actual_domain:
                            has_mismatch_link_text = 1
                    except Exception:
                        has_mismatch_link_text = 1

        return {
            "num_links": num_links,
            "num_external_links": num_external,
            "num_internal_links": num_internal,
            "num_null_self_links": num_null_self,
            "ratio_external_links": round(num_external / num_links, 4),
            "ratio_internal_links": round(num_internal / num_links, 4),
            "ratio_null_self_links": round(num_null_self / num_links, 4),
            "num_suspicious_anchor_text": num_suspicious_anchor_text,
            "has_mismatch_link_text": has_mismatch_link_text,
        }

    @staticmethod
    def extract_security_script_features(soup: BeautifulSoup) -> dict[str, int]:
        """Extract anti-analysis, iframe, popup, and JavaScript obfuscation signals.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content of the page.

        Returns
        -------
        dict[str, int]
            Dictionary of security and script features.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        has_right_click_disabled = 0
        has_text_selection_disabled = 0
        num_hidden_iframes = 0
        has_popup_script = 0
        has_obfuscated_js = 0

        script_texts: list[str] = []
        for script in soup.find_all("script"):
            if script.string:
                script_texts.append(script.string)
            elif script.contents:
                script_texts.append(" ".join(str(c) for c in script.contents))

        all_scripts_combined = " ".join(script_texts).lower()

        inline_handlers: list[str] = []
        for tag in soup.find_all(True):
            for attr, val in tag.attrs.items():
                if attr.lower().startswith("on"):
                    val_str = str(val).lower()
                    inline_handlers.append(f"{attr}={val_str}")
                    if attr.lower() in ("oncontextmenu", "oncontext"):
                        has_right_click_disabled = 1
                    if attr.lower() in ("onselectstart", "ondragstart"):
                        has_text_selection_disabled = 1

        all_handlers_combined = " ".join(inline_handlers)

        # Right click disable check
        if not has_right_click_disabled:
            if re.search(r"oncontextmenu\s*=", all_scripts_combined) or \
               (re.search(r"contextmenu", all_scripts_combined) and ("preventdefault" in all_scripts_combined or "return false" in all_scripts_combined)) or \
               "event.button == 2" in all_scripts_combined or "event.button==2" in all_scripts_combined:
                has_right_click_disabled = 1

        # Text selection disable check
        if not has_text_selection_disabled:
            if "user-select: none" in all_scripts_combined or "user-select:none" in all_scripts_combined or \
               "onselectstart" in all_scripts_combined or "onselectstart" in all_handlers_combined:
                has_text_selection_disabled = 1

            if not has_text_selection_disabled:
                for tag in soup.find_all(True):
                    style_attr = str(tag.get("style", "")).lower()
                    if "user-select" in style_attr and "none" in style_attr:
                        has_text_selection_disabled = 1
                        break
            if not has_text_selection_disabled:
                for style_tag in soup.find_all("style"):
                    if "user-select" in style_tag.get_text().lower() and "none" in style_tag.get_text().lower():
                        has_text_selection_disabled = 1
                        break

        # iFrame analysis
        iframes = soup.find_all("iframe")
        num_iframes = len(iframes)

        for iframe in iframes:
            style = str(iframe.get("style", "")).lower().replace(" ", "")
            width = str(iframe.get("width", "")).strip().lower()
            height = str(iframe.get("height", "")).strip().lower()
            is_hidden_attr = iframe.has_attr("hidden")

            if is_hidden_attr or \
               "display:none" in style or "visibility:hidden" in style or "opacity:0" in style or \
               "width:0" in style or "height:0" in style or "width:1px" in style or "height:1px" in style or \
               width in ("0", "0px", "1", "1px") or height in ("0", "0px", "1", "1px"):
                num_hidden_iframes += 1

        # Popup script check
        if re.search(r"\bwindow\.open\s*\(", all_scripts_combined) or \
           re.search(r"\balert\s*\(", all_scripts_combined) or \
           re.search(r"\bprompt\s*\(", all_scripts_combined) or \
           re.search(r"\bconfirm\s*\(", all_scripts_combined) or \
           "window.open" in all_handlers_combined or "alert(" in all_handlers_combined:
            has_popup_script = 1

        # JS Obfuscation check
        obfuscation_patterns = [
            r"\beval\s*\(",
            r"\bunescape\s*\(",
            r"\bString\.fromCharCode\s*\(",
            r"\batob\s*\(",
            r"(?:\\x[0-9a-fA-F]{2}){4,}",
            r"(?:\\u[0-9a-fA-F]{4}){4,}",
        ]
        for pattern in obfuscation_patterns:
            if re.search(pattern, all_scripts_combined):
                has_obfuscated_js = 1
                break

        return {
            "has_right_click_disabled": has_right_click_disabled,
            "has_text_selection_disabled": has_text_selection_disabled,
            "num_iframes": num_iframes,
            "num_hidden_iframes": num_hidden_iframes,
            "has_popup_script": has_popup_script,
            "has_obfuscated_js": has_obfuscated_js,
        }

    @staticmethod
    def extract_metadata_dom_features(soup: BeautifulSoup, page_url: str = "") -> dict[str, int]:
        """Extract metadata headers, favicon origin, meta-redirect, and DOM tree structural metrics.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content of the page.
        page_url : str, optional
            The URL of the rendered page, used to evaluate domain matching and external favicons.

        Returns
        -------
        dict[str, Any]
            Dictionary of metadata and DOM structural signals.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        page_url = page_url or ""
        identity = parse_host_identity(page_url) if isinstance(page_url, str) else parse_host_identity("")
        page_host = identity["host"]
        page_domain = identity["registered_domain"]
        brand_name = identity["brand_name"]
        _tld_ext_domain = identity["domain"]
        _tld_ext_suffix = identity["suffix"]
        _tld_ext_registered = identity["registered_domain"]

        # External Favicon Check
        has_external_favicon = 0
        icon_links = soup.find_all("link", rel=lambda r: r and any("icon" in str(item).lower() for item in (r if isinstance(r, list) else [r])))
        for link in icon_links:
            href = (link.get("href") or "").strip()
            if href:
                try:
                    parsed_href = urlparse(href)
                    href_host = (parsed_href.hostname or "").lower()
                    if href_host:
                        href_domain = parse_host_identity(href_host)["registered_domain"] or href_host

                        if page_domain and href_domain != page_domain:
                            has_external_favicon = 1
                            break
                        elif not page_domain and page_host and href_host != page_host:
                            has_external_favicon = 1
                            break
                except Exception:
                    pass

        # Meta Refresh Check
        has_meta_refresh = 0
        meta_tags = soup.find_all("meta")
        num_meta_tags = len(meta_tags)

        for meta in meta_tags:
            http_equiv = str(meta.get("http-equiv", "")).strip().lower()
            if http_equiv == "refresh":
                has_meta_refresh = 1
                break

        description = soup.find("meta", attrs={"name": lambda value: isinstance(value, str) and value.lower() == "description"})
        meta_description = description.get("content", "").strip() if description else ""

        # Title Matches Domain Check
        raw_title = getattr(HTMLFeatureExtractor, '_last_page_title', None)
        if not raw_title and soup.title:
            raw_title = soup.title.get_text(" ", strip=True)
        raw_title = raw_title or ""

        import re
        def normalize_str(s):
            if not s: return ""
            s = s.lower()
            s = re.sub(r'[|,;:]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        normalized_title = normalize_str(raw_title)
        normalized_description = normalize_str(meta_description)
        searchable_text = f"{normalized_title} {normalized_description}".strip()

        brand_candidates: list[str] = []
        for token in (brand_name, identity.get("domain", ""), page_domain.split(".")[0] if page_domain else ""):
            token_n = normalize_str(token)
            if token_n and token_n not in _SUFFIX_LIKE_LABELS and token_n not in brand_candidates:
                brand_candidates.append(token_n)
        for label in (identity.get("host") or "").split("."):
            label_n = normalize_str(label)
            if label_n and label_n not in {"www"} and label_n not in _SUFFIX_LIKE_LABELS and label_n not in brand_candidates:
                brand_candidates.append(label_n)
        # Prefer longer brand tokens so "kongu" wins over a leftover "ac".
        brand_candidates.sort(key=len, reverse=True)

        title_matches_domain = 0
        title_domain_similarity_score = None
        match_tier_used = "None"
        normalized_brand = brand_candidates[0] if brand_candidates else normalize_str(brand_name)

        if brand_candidates and searchable_text:
            import json
            import os
            try:
                aliases_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "brand_aliases.json")
                with open(aliases_path, "r", encoding="utf-8") as f:
                    brand_aliases = json.load(f)
            except Exception:
                brand_aliases = {}

            matched = False
            for candidate in brand_candidates:
                aliases_for_brand = brand_aliases.get(candidate, [])
                if candidate == normalized_title:
                    title_matches_domain = 1
                    title_domain_similarity_score = 100.0
                    match_tier_used = "Exact Match"
                    normalized_brand = candidate
                    matched = True
                    break
                alias_hit = False
                for alias in aliases_for_brand:
                    alias_norm = normalize_str(alias)
                    if alias_norm and (alias_norm in searchable_text or searchable_text in alias_norm):
                        title_matches_domain = 1
                        title_domain_similarity_score = 100.0
                        match_tier_used = f"Alias Match ({alias})"
                        normalized_brand = candidate
                        alias_hit = True
                        matched = True
                        break
                if alias_hit:
                    break
                if candidate in searchable_text:
                    title_matches_domain = 1
                    title_domain_similarity_score = 100.0
                    match_tier_used = "Substring Match"
                    normalized_brand = candidate
                    matched = True
                    break

            if not matched:
                import difflib
                tokens = searchable_text.split()
                best_token_ratio = 0.0
                best_candidate = normalized_brand
                for candidate in brand_candidates:
                    for token in tokens:
                        if len(token) < 3:
                            continue
                        ratio = difflib.SequenceMatcher(None, candidate, token).ratio()
                        if ratio > best_token_ratio:
                            best_token_ratio = ratio
                            best_candidate = candidate
                    fallback_ratio = difflib.SequenceMatcher(None, candidate, searchable_text).ratio()
                    if fallback_ratio > best_token_ratio:
                        best_token_ratio = fallback_ratio
                        best_candidate = candidate
                title_domain_similarity_score = round(best_token_ratio * 100.0, 2)
                match_tier_used = "Token Similarity"
                normalized_brand = best_candidate

        # DOM Structural Metrics (Total Elements & Tree Depth)

        all_elements = soup.find_all(True)
        num_total_dom_elements = len(all_elements)

        def _calc_depth(element, current_depth=1) -> int:
            children = [child for child in element.children if getattr(child, "name", None) is not None]
            if not children:
                return current_depth
            return max(_calc_depth(child, current_depth + 1) for child in children)

        root = soup.find("html") or soup
        dom_depth = _calc_depth(root) if root else 0

        _raw_return = {
            "has_external_favicon": has_external_favicon,
            "has_meta_refresh": has_meta_refresh,
            "title_matches_domain": title_matches_domain,
            "title_domain_similarity_score": title_domain_similarity_score,
            "num_meta_tags": num_meta_tags,
            "dom_depth": dom_depth,
            "num_total_dom_elements": num_total_dom_elements,
            "page_title": raw_title,
            "meta_description": meta_description,
            "extracted_brand_name": brand_name,
            "extracted_page_domain": page_domain,
            "extracted_registered_domain": _tld_ext_registered,
            "extracted_tld_domain": _tld_ext_domain,
            "extracted_tld_suffix": _tld_ext_suffix,
        }

        return _raw_return

    @classmethod
    def extract_all_html_features(cls, soup: BeautifulSoup, page_url: str = "") -> dict[str, Any]:
        """Consolidate features from all HTML submodules into a single dictionary.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content of the page.
        page_url : str, optional
            The URL of the rendered page, used to evaluate relative vs. external elements.

        Returns
        -------
        dict[str, Any]
            Unified dictionary of all HTML-extracted features.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        combined: dict[str, Any] = {}
        combined.update(cls.extract_basic_information(soup))
        combined.update(cls.extract_form_features(soup, page_url=page_url))
        combined.update(cls.extract_link_features(soup, page_url=page_url))
        combined.update(cls.extract_security_script_features(soup))
        combined.update(cls.extract_metadata_dom_features(soup, page_url=page_url))
        return combined

    def close_browser(self) -> None:
        """Close context, Chromium, and Playwright resources; safe to call repeatedly."""
        for resource_name, resource, closer in (
            ("context", self._context, "close"),
            ("browser", self._browser, "close"),
            ("playwright", self._playwright, "stop"),
        ):
            if resource is None:
                continue
            try:
                getattr(resource, closer)()
            except Exception:
                # Cleanup must never hide the browser-initialization exception.
                try:
                    _LOG.exception("Failed to close Playwright %s", resource_name)
                except (OSError, ValueError):
                    pass
        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(self) -> "HTMLFeatureExtractor":
        self.launch_browser()
        return self

    def __exit__(self, *_: object) -> None:
        self.close_browser()


def extract_hybrid_features(url: str, html_extractor: HTMLFeatureExtractor | None = None) -> pd.DataFrame:
    """Extract both URL lexical/network features and rendered-HTML content features.

    Parameters
    ----------
    url : str
        Target HTTP or HTTPS web address to analyze.
    html_extractor : HTMLFeatureExtractor, optional
        An existing HTMLFeatureExtractor instance with active Chromium context. If None,
        a temporary context will be launched and closed automatically.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing combined URL and HTML feature vectors.
    """
    started = time.perf_counter()

    # 1. URL Feature Extraction
    url_frame = None
    try:
        from utils.url_feature_extractor import extract_all_features as extract_url_all
        url_frame = extract_url_all(url)
    except Exception:
        url_frame = pd.DataFrame([{"url": url}])

    # 2. HTML Feature Extraction
    html_dict: dict[str, Any] = {}
    if html_extractor is not None:
        html_str = html_extractor.fetch_rendered_html(url)
        soup = html_extractor.parse_html(html_str)
        html_dict = html_extractor.extract_all_html_features(soup, page_url=url)
    else:
        with HTMLFeatureExtractor() as extractor:
            html_str = extractor.fetch_rendered_html(url)
            soup = extractor.parse_html(html_str)
            html_dict = extractor.extract_all_html_features(soup, page_url=url)

    # 3. Combine into unified DataFrame
    html_frame = pd.DataFrame([html_dict])
    combined_frame = pd.concat([url_frame.reset_index(drop=True), html_frame.reset_index(drop=True)], axis=1)
    combined_frame.attrs.update({
        "url": url,
        "extraction_seconds": time.perf_counter() - started,
    })
    return combined_frame
