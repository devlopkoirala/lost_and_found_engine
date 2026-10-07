from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10

st.set_page_config(
    page_title="Lost & Found Visual Search Engine",
    page_icon="🔎",
    layout="wide",
)

st.title("Lost & Found Visual Search Engine")
st.write(
    "Upload a photo of a lost item to find visually similar items in the catalog."
)

with st.sidebar:
    st.header("Search settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).strip().rstrip("/")
    top_k = st.slider("Results to show (top_k)", min_value=1, max_value=10, value=5)
    st.divider()
    st.subheader("Backend status")

    if not api_url:
        st.error("Enter an API URL to check backend health.")
    else:
        try:
            health_response = requests.get(
                f"{api_url}/",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            health_response.raise_for_status()
            health_data: Any = health_response.json()
            if not isinstance(health_data, dict) or health_data.get("status") != "online":
                st.warning("The API responded, but did not report an online status.")
            else:
                st.success("Backend online")
                catalog_items = health_data.get("catalog_items")
                if isinstance(catalog_items, int):
                    st.caption(f"Indexed catalog items: {catalog_items}")
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")
        except ValueError as exc:
            st.error(f"Backend returned invalid health-check JSON: {exc}")

query_image = st.file_uploader(
    "Choose a query photo",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, and PNG.",
)

if query_image is not None:
    st.image(query_image, caption=f"Query image: {query_image.name}", width=300)

if st.button("Search catalog", type="primary", disabled=query_image is None):
    if not api_url:
        st.error("Enter an API URL in the sidebar before searching.")
    elif query_image is None:
        st.error("Choose an image before searching.")
    else:
        try:
            response = requests.post(
                f"{api_url}/search",
                params={"top_k": top_k},
                files={
                    "file": (
                        query_image.name,
                        query_image.getvalue(),
                        query_image.type or "application/octet-stream",
                    )
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not response.ok:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", response.text)
                except ValueError:
                    detail = response.text or response.reason
                st.error(f"Search failed ({response.status_code}): {detail}")
            else:
                try:
                    result: Any = response.json()
                except ValueError as exc:
                    st.error(f"Backend returned invalid search JSON: {exc}")
                else:
                    matches = result.get("matches") if isinstance(result, dict) else None
                    if not isinstance(matches, list):
                        st.error("Backend response is missing a valid 'matches' list.")
                    else:
                        st.subheader("Closest matches")
                        st.caption(
                            f"{result.get('total_results', len(matches))} result(s) returned"
                        )
                        if not matches:
                            st.info("No matching catalog items were found.")
                        else:
                            columns = st.columns(2)
                            for index, match in enumerate(matches):
                                if not isinstance(match, dict):
                                    st.warning("Skipped a malformed match in the API response.")
                                    continue

                                image_path = match.get("image_path")
                                score = match.get("similarity_score")
                                column = columns[index % len(columns)]
                                with column:
                                    st.markdown(f"**Match {index + 1}**")
                                    if isinstance(image_path, str):
                                        local_path = (PROJECT_ROOT / image_path).resolve()
                                        try:
                                            local_path.relative_to(PROJECT_ROOT)
                                        except ValueError:
                                            st.warning("Result image path is outside the project.")
                                        else:
                                            if local_path.is_file():
                                                st.image(local_path, use_container_width=True)
                                            else:
                                                st.info("Catalog image is not available locally.")
                                        st.caption(image_path)
                                    if isinstance(score, (int, float)):
                                        st.metric("Cosine similarity", f"{score:.4f}")
        except requests.RequestException as exc:
            st.error(f"Could not complete the search request: {exc}")
