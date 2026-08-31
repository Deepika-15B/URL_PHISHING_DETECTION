"""Standalone rendered-HTML feature extraction for Phase 3.

This module deliberately does not load a phishing model or perform prediction.
"""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

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


class CDNErrorPageError(HTMLFeatureExtractionError):
    """Raised when a CDN error page is detected instead of the actual content."""



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
            self._browser = self._playwright.chromium.launch(headless=True)
            print("Browser launched")
            self._context = self._browser.new_context(
                java_script_enabled=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
                color_scheme="light",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1"
                }
            )
            self._context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                window.navigator.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)
            print("Context created with stealth settings")
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
            print("Page created")
            page.set_default_timeout(self.timeout_ms)
            # Retry loop for navigation
            for attempt in range(2):
                try:
                    try:
                        print(f"URL navigation started (networkidle) - attempt {attempt + 1}")
                        page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
                        print("Navigation completed")
                    except PlaywrightTimeoutError:
                        print("URL navigation started (fallback)")
                        page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                        page.wait_for_timeout(750)
                        print("Navigation completed (fallback)")
                    break
                except PlaywrightTimeoutError as error:
                    if attempt == 0:
                        print(f"PlaywrightTimeoutError on attempt 1, retrying in 5 seconds...")
                        page.wait_for_timeout(5000)
                    else:
                        raise error
                except Exception as error:
                    if attempt == 0:
                        print(f"Navigation error ({error}) on attempt 1, retrying in 5 seconds...")
                        page.wait_for_timeout(5000)
                    else:
                        raise error
            
            def is_anti_bot(title_text: str, content_text: str) -> bool:
                t_lower = title_text.lower()
                c_lower = content_text.lower()
                if "just a moment" in t_lower or "attention required" in t_lower or "security check" in t_lower:
                    return True
                if "cf-browser-verification" in c_lower or "challenge-platform" in c_lower or "incapsula" in c_lower or "sucuri_cloudproxy" in c_lower or "akamai bot manager" in c_lower:
                    return True
                return False

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

            if is_anti_bot(title, content):
                print("Anti-bot detected, waiting 8 seconds for challenge to resolve...")
                page.wait_for_timeout(8000)
                content = page.content()
                title = page.title()
                if is_anti_bot(title, content):
                    raise AntiBotProtectionError("Cloudflare/Anti-bot challenge detected. Cannot analyze.")

            if is_cdn_error(title, content):
                print("CDN Error Page detected.")
                raise CDNErrorPageError("CDN/CloudFront error page detected. Actual webpage could not be analysed.")

            print("========== PAGE INFO ==========")
            print("page.url() =", page.url)
            print("page.title() =", title)
            print("===============================")
            HTMLFeatureExtractor._last_page_url   = page.url
            HTMLFeatureExtractor._last_page_title = page.title()
            return content
        except PlaywrightTimeoutError as error:
            import traceback
            traceback.print_exc()
            raise HTMLFeatureExtractionError(f"Timed out after {self.timeout_ms} ms while loading {url!r}.") from error
        except (AntiBotProtectionError, CDNErrorPageError) as error:
            raise error
        except Exception as error:
            import traceback
            traceback.print_exc()
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
        page_host = ""
        page_domain = ""
        if page_url and isinstance(page_url, str):
            try:
                parsed_page = urlparse(page_url.strip())
                page_host = (parsed_page.hostname or "").lower()
                parts = page_host.split(".")
                page_domain = ".".join(parts[-2:]) if len(parts) >= 2 else page_host
            except Exception:
                page_host = ""
                page_domain = ""

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
                    action_parts = action_host.split(".")
                    action_domain = ".".join(action_parts[-2:]) if len(action_parts) >= 2 else action_host

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

        page_host = ""
        page_domain = ""
        if page_url and isinstance(page_url, str):
            try:
                parsed_page = urlparse(page_url.strip())
                page_host = (parsed_page.hostname or "").lower()
                parts = page_host.split(".")
                page_domain = ".".join(parts[-2:]) if len(parts) >= 2 else page_host
            except Exception:
                page_host = ""
                page_domain = ""

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
                        href_parts = href_host.split(".")
                        href_domain = ".".join(href_parts[-2:]) if len(href_parts) >= 2 else href_host

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
                text_host = match.group(1).lower()
                text_parts = text_host.split(".")
                text_domain = ".".join(text_parts[-2:]) if len(text_parts) >= 2 else text_host

                if is_null_self or is_external:
                    has_mismatch_link_text = 1
                elif is_internal and href:
                    try:
                        parsed_href = urlparse(href)
                        actual_host = (parsed_href.hostname or page_host).lower()
                        actual_parts = actual_host.split(".")
                        actual_domain = ".".join(actual_parts[-2:]) if len(actual_parts) >= 2 else actual_host
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

        page_host = ""
        page_domain = ""
        brand_name = ""
        _tld_ext_domain = ""
        _tld_ext_suffix = ""
        _tld_ext_registered = ""
        _tld_exception = None
        page_url = page_url or ""
        if page_url and isinstance(page_url, str):
            try:
                import tldextract
                ext = tldextract.extract(page_url)
                _tld_ext_domain     = ext.domain
                _tld_ext_suffix     = ext.suffix
                _tld_ext_registered = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
                if ext.domain:
                    brand_name = ext.domain
                    page_domain = _tld_ext_registered
                page_host = urlparse(page_url.strip()).hostname or ""
            except Exception as _e:
                _tld_exception = _e
                page_host = urlparse(page_url.strip()).hostname or ""
                parts = page_host.split('.')
                page_domain = ".".join(parts[-2:]) if len(parts) >= 2 else page_host
                brand_name = parts[-2] if len(parts) >= 2 else page_host

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
                        href_parts = href_host.split(".")
                        href_domain = ".".join(href_parts[-2:]) if len(href_parts) >= 2 else href_host

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
        normalized_brand = normalize_str(brand_name)

        title_matches_domain = 0
        title_domain_similarity_score = None
        match_tier_used = "None"

        if normalized_brand and normalized_title:
            # Load aliases
            import json
            import os
            try:
                aliases_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "brand_aliases.json")
                with open(aliases_path, "r", encoding="utf-8") as f:
                    brand_aliases = json.load(f)
            except Exception:
                brand_aliases = {}
                
            aliases_for_brand = brand_aliases.get(normalized_brand, [])
            
            # 1. Exact match
            if normalized_brand == normalized_title:
                title_matches_domain = 1
                title_domain_similarity_score = 100.0
                match_tier_used = "Exact Match"
            else:
                # 2. Alias match
                alias_matched = False
                for alias in aliases_for_brand:
                    alias_norm = normalize_str(alias)
                    if alias_norm and (alias_norm in normalized_title or normalized_title in alias_norm):
                        title_matches_domain = 1
                        title_domain_similarity_score = 100.0
                        alias_matched = True
                        match_tier_used = f"Alias Match ({alias})"
                        break
                
                if not alias_matched:
                    # 3. Substring match
                    if normalized_brand in normalized_title:
                        title_matches_domain = 1
                        title_domain_similarity_score = 100.0
                        match_tier_used = "Substring Match"
                    else:
                        import difflib
                        # 4. Token similarity
                        tokens = normalized_title.split()
                        best_token_ratio = 0.0
                        for token in tokens:
                            ratio = difflib.SequenceMatcher(None, normalized_brand, token).ratio()
                            if ratio > best_token_ratio:
                                best_token_ratio = ratio
                                
                        # 5. SequenceMatcher fallback (entire title)
                        fallback_ratio = difflib.SequenceMatcher(None, normalized_brand, normalized_title).ratio()
                        
                        if best_token_ratio >= fallback_ratio:
                            title_domain_similarity_score = round(best_token_ratio * 100.0, 2)
                            match_tier_used = "Token Similarity"
                        else:
                            title_domain_similarity_score = round(fallback_ratio * 100.0, 2)
                            match_tier_used = "SequenceMatcher Fallback"

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

        description = soup.find("meta", attrs={"name": lambda value: isinstance(value, str) and value.lower() == "description"})
        meta_description = description.get("content", "").strip() if description else ""

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
        }

        return _raw_return

    @staticmethod
    def compute_domain_title_match_score(page_url: str, raw_title: str) -> float:
        """Calculate DomainTitleMatchScore using the existing similarity implementation.
        
        Isolates similarity calculation for dataset-level validation.
        """
        if not page_url or not raw_title:
            return 0.0

        import re
        def normalize_str(s: str) -> str:
            if not s:
                return ""
            s = s.lower()
            s = re.sub(r'[|,;:\-_/]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        brand_name = ""
        try:
            import tldextract
            ext = tldextract.extract(page_url)
            if ext.domain:
                brand_name = ext.domain.lower()
        except Exception:
            pass

        if not brand_name:
            try:
                from urllib.parse import urlparse
                host = (urlparse(page_url.strip()).hostname or "").lower()
                parts = host.split('.')
                brand_name = parts[-2] if len(parts) >= 2 else host
            except Exception:
                pass

        normalized_title = normalize_str(raw_title)
        normalized_brand = normalize_str(brand_name)

        if not normalized_brand or not normalized_title:
            return 0.0

        if normalized_brand in normalized_title:
            return 100.0

        import difflib
        match = difflib.SequenceMatcher(None, normalized_brand, normalized_title).find_longest_match(0, len(brand_name), 0, len(normalized_title))
        if brand_name:
            return round((match.size / len(brand_name)) * 100.0, 2)
        return 0.0

    @classmethod
    def extract_phiusiil_html_features(cls, soup: BeautifulSoup, raw_html: str, page_url: str = "") -> dict[str, Any]:
        """Extract the specific 13 HTML features required by the 18 leakage-free feature set.

        Features extracted:
        - LineOfCode: Count lines in retrieved raw HTML source using newline splitting.
        - LargestLineLength: Maximum character length of a raw HTML line.
        - NoOfImage: Count of image elements (<img tags>).
        - NoOfJS: Count of external JavaScript files (<script src=...>).
        - NoOfCSS: Count of external CSS files (<link rel="stylesheet">).
        - HasDescription: 1 if meta description exists and is non-empty, otherwise 0.
        - IsResponsive: 1 if viewport meta tag exists, otherwise 0.
        - HasSubmitButton: 1 if at least one submit input/button exists, otherwise 0.
        - HasSocialNet: 1 if page contains links/resources belonging to common social networks, otherwise 0.
        - HasCopyrightInfo: 1 if page text or HTML contains ©, &copy;, &#169;, or "copyright", otherwise 0.
        - NoOfExternalRef: Count of external resource references across HTML tags.
        - NoOfSelfRef: Count of internal/self references across HTML tags.
        - DomainTitleMatchScore: Substring/SequenceMatcher similarity between domain brand name and page title.
        """
        if soup is None:
            raise ValueError("A parsed BeautifulSoup object is required.")

        raw_html_str = raw_html if isinstance(raw_html, str) else ""

        # 1. LineOfCode: count lines in the retrieved raw HTML source using newline splitting
        lines = raw_html_str.split('\n') if raw_html_str else []
        line_of_code = len(lines)

        # 2. LargestLineLength: calculate maximum character length of a raw HTML line
        largest_line_length = max((len(line) for line in lines), default=0)

        # 3. NoOfImage: map from existing image count
        no_of_image = len(soup.find_all("img"))

        # 4. NoOfJS: map from existing JavaScript file count (script tags with src attribute)
        no_of_js = len(soup.find_all("script", src=True))

        # 5. NoOfCSS: map from existing CSS file count (link tags with rel stylesheet)
        def _is_stylesheet(rel_val):
            if not rel_val:
                return False
            if isinstance(rel_val, list):
                rel_val = " ".join(rel_val)
            return "stylesheet" in str(rel_val).lower()

        no_of_css = len(soup.find_all("link", rel=_is_stylesheet))

        # 6. HasDescription: return 1 if a meta description exists and is non-empty, otherwise 0
        desc_tag = soup.find("meta", attrs={"name": lambda v: isinstance(v, str) and v.lower() == "description"})
        meta_desc = desc_tag.get("content", "").strip() if desc_tag and desc_tag.get("content") else ""
        has_description = 1 if bool(meta_desc) else 0

        # 7. IsResponsive: return 1 if a viewport meta tag exists, otherwise 0
        viewport_tag = soup.find("meta", attrs={"name": lambda v: isinstance(v, str) and v.lower() == "viewport"})
        is_responsive = 1 if viewport_tag is not None else 0

        # 8. HasSubmitButton: return 1 if at least one submit input/button exists, otherwise 0
        has_submit_button = 0
        for inp in soup.find_all("input"):
            itype = str(inp.get("type", "")).strip().lower()
            if itype in ("submit", "image"):
                has_submit_button = 1
                break
        if not has_submit_button:
            for btn in soup.find_all("button"):
                btype = str(btn.get("type", "")).strip().lower()
                if btype in ("submit", ""):
                    has_submit_button = 1
                    break

        # 9. HasSocialNet: return 1 if page contains links/resources belonging to common social-network domains; otherwise 0
        has_social_net = 0
        social_domains = {
            "facebook.com", "fb.com", "twitter.com", "x.com", "linkedin.com",
            "instagram.com", "youtube.com", "pinterest.com", "tiktok.com",
            "reddit.com", "snapchat.com", "whatsapp.com", "t.me", "telegram.org",
            "tumblr.com", "threads.net"
        }
        for tag in soup.find_all(["a", "link", "script", "img", "iframe", "source", "embed"]):
            url_attr = tag.get("href") or tag.get("src")
            if url_attr:
                try:
                    from urllib.parse import urlparse
                    host = (urlparse(str(url_attr).strip()).hostname or "").lower()
                    if host and any(host == sd or host.endswith("." + sd) for sd in social_domains):
                        has_social_net = 1
                        break
                except Exception:
                    pass

        # 10. HasCopyrightInfo: return 1 if page text contains ©, &copy;, &#169;, or "copyright"; otherwise 0
        has_copyright_info = 0
        text_content = soup.get_text().lower() if hasattr(soup, "get_text") else ""
        raw_html_lower = raw_html_str.lower()
        if "©" in text_content or "©" in raw_html_lower or "&copy;" in raw_html_lower or "&#169;" in raw_html_lower or "copyright" in text_content:
            has_copyright_info = 1

        # Determine host and registered domain of the page
        page_domain = ""
        page_host = ""
        if page_url and isinstance(page_url, str):
            try:
                import tldextract
                ext = tldextract.extract(page_url.strip())
                page_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
                from urllib.parse import urlparse
                page_host = (urlparse(page_url.strip()).hostname or "").lower()
            except Exception:
                try:
                    from urllib.parse import urlparse
                    page_host = (urlparse(page_url.strip()).hostname or "").lower()
                    parts = page_host.split(".")
                    page_domain = ".".join(parts[-2:]) if len(parts) >= 2 else page_host
                except Exception:
                    pass

        # 11. NoOfExternalRef & 12. NoOfSelfRef:
        # Count resource references across relevant HTML tags: a, link, script, img, iframe, form, video, audio, source, embed, object
        # Definition:
        # - Self/Internal: Relative URL, fragment (#), inline protocol (javascript:, about:blank, mailto:, tel:, data:), or URL on the same registered domain/host.
        # - External: Resource located on a domain different from the page's registered domain.
        no_of_external_ref = 0
        no_of_self_ref = 0

        for tag in soup.find_all(["a", "link", "script", "img", "iframe", "form", "video", "audio", "source", "embed", "object"]):
            url_attr = tag.get("href") or tag.get("src") or tag.get("action") or tag.get("data")
            if url_attr is not None:
                url_str = str(url_attr).strip()
                url_lower = url_str.lower()
                if not url_str or url_str == "#" or url_lower.startswith(("javascript:", "about:blank", "mailto:", "tel:", "data:", "#")):
                    no_of_self_ref += 1
                else:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url_str)
                        host = (parsed.hostname or "").lower()
                        if not parsed.scheme and not host:
                            no_of_self_ref += 1
                        elif host:
                            try:
                                import tldextract
                                ref_ext = tldextract.extract(host)
                                ref_domain = f"{ref_ext.domain}.{ref_ext.suffix}" if ref_ext.suffix else ref_ext.domain
                            except Exception:
                                parts = host.split(".")
                                ref_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host

                            if page_domain and ref_domain == page_domain:
                                no_of_self_ref += 1
                            elif not page_domain and page_host and host == page_host:
                                no_of_self_ref += 1
                            else:
                                no_of_external_ref += 1
                        else:
                            no_of_self_ref += 1
                    except Exception:
                        no_of_self_ref += 1

        # 13. DomainTitleMatchScore:
        raw_title = getattr(cls, '_last_page_title', None)
        if not raw_title and soup.title:
            raw_title = soup.title.get_text(" ", strip=True)
        raw_title = raw_title or ""

        domain_title_match_score = cls.compute_domain_title_match_score(page_url, raw_title)

        return {
            "LineOfCode": line_of_code,
            "LargestLineLength": largest_line_length,
            "NoOfImage": no_of_image,
            "NoOfJS": no_of_js,
            "NoOfCSS": no_of_css,
            "HasDescription": has_description,
            "IsResponsive": is_responsive,
            "HasSubmitButton": has_submit_button,
            "HasSocialNet": has_social_net,
            "HasCopyrightInfo": has_copyright_info,
            "NoOfExternalRef": no_of_external_ref,
            "NoOfSelfRef": no_of_self_ref,
            "DomainTitleMatchScore": float(domain_title_match_score),
        }

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

        all_features = {}
        all_features.update(cls.extract_basic_information(soup))
        all_features.update(cls.extract_form_features(soup, page_url))
        all_features.update(cls.extract_link_features(soup, page_url))
        all_features.update(cls.extract_security_script_features(soup))
        all_features.update(cls.extract_metadata_dom_features(soup, page_url))
        return all_features

    def close_browser(self) -> None:

        """Close context, Chromium, and Playwright resources; safe to call repeatedly."""
        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = None
            try:
                if self._browser is not None:
                    self._browser.close()
                    print("Browser closed")
            finally:
                self._browser = None
                if self._playwright is not None:
                    self._playwright.stop()
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
