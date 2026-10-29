"""Make NEMAR downloads survive a slow or flaky data.nemar.org.

neuralfetch calls ``nemar.download`` with the defaults of nemar-py 0.3.3: 30 s
per metadata request and 5 retries, which fail when the manifest server is
slow. ``patch()`` raises the timeouts, retries the whole download a few times
(files already on disk are skipped, so a rerun resumes), and fetches the
manifest from its public S3 copy when data.nemar.org does not answer.
Call it once before ``Study(...).download()``.
"""

import functools
import random
import time


def patch(outer_attempts=6, metadata_timeout=180.0):
    try:
        import httpx
        import nemar
        from nemar._client import NEMARClient
        from nemar._models import VersionManifest
        from nemar.errors import TransferError, TransportError
    except ImportError:  # another nemar-py version: leave it as it is
        return
    if getattr(nemar.download, "_patient", False):
        return

    fetch_manifest = NEMARClient.fetch_manifest

    def fetch_manifest_with_s3_copy(self, dataset_index, version):
        try:
            return fetch_manifest(self, dataset_index, version)
        except TransportError as error:
            tag = version.version if version.version.startswith("v") else f"v{version.version}"
            url = f"https://nemar.s3.us-east-2.amazonaws.com/{dataset_index.dataset_id}/version/{tag}.json"
            print(f"[nemar] manifest unavailable ({error}); using its S3 copy {url}")
            response = httpx.get(url, timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True)
            response.raise_for_status()
            return VersionManifest.parse(response.json(), manifest_url=url, endpoint=self.endpoint)

    download = nemar.download

    @functools.wraps(download)
    def patient_download(*args, **kwargs):
        kwargs.setdefault("metadata_timeout", metadata_timeout)
        kwargs.setdefault("max_retries", 8)
        kwargs.setdefault("stream_timeout", 300.0)
        for attempt in range(outer_attempts):
            try:
                return download(*args, **kwargs)
            except (TransportError, TransferError, httpx.TransportError) as error:
                if attempt == outer_attempts - 1:
                    raise
                wait = min(600.0, 30.0 * 2 ** attempt) * (1 + random.random() / 2)
                print(f"[nemar] attempt {attempt + 1}/{outer_attempts} failed: {error}; retrying in {wait:.0f} s")
                time.sleep(wait)

    patient_download._patient = True
    NEMARClient.fetch_manifest = fetch_manifest_with_s3_copy
    nemar.download = patient_download  # neuralfetch looks it up at call time
