import sys
sys.path.insert(0, '.')
from utils.html_feature_extractor import HTMLFeatureExtractor

extractor = HTMLFeatureExtractor()
extractor.launch_browser()
urls = [
    'https://www.onlinesbi.sbi',
    'https://www.tn.gov.in',
    'https://kongu.ac.in',
]
for url in urls:
    html = extractor.fetch_rendered_html(url)
    soup = extractor.parse_html(html)
    raw = extractor.extract_all_html_features(soup, page_url=url)
    print(f"URL: {url}")
    print(f"  page_title: {raw.get('page_title')}")
    print(f"  extracted_brand_name: {raw.get('extracted_brand_name')}")
    print(f"  title_domain_similarity_score: {raw.get('title_domain_similarity_score')}")
    print()
extractor.close_browser()
