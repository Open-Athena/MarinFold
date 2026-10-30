"""Read a verified regional mirror while retaining the original data cursor.

CorpusStream seeds row ordering with the original URI and binds its cursor to
the original catalog hash. Rewriting that catalog during a GPU-to-TPU fork
would silently change the continuation. A separate physical-location map keeps
those identities intact. The staging process must verify content equality and
record destination metadata before publishing the map; this class does not copy
data or fall back to the original region when a mirror is absent.
"""

import copy
from typing import Any

from transformers import PreTrainedTokenizerBase

from corpus_stream import CorpusStream


class MirroredCorpusStream(CorpusStream):
    """Continue a logical data rank from an entirely local, verified mirror."""

    def __init__(
        self,
        prefix: str,
        manifest: dict,
        tokenizer: PreTrainedTokenizerBase,
        rank: int,
        world: int,
        mirror: dict,
    ) -> None:
        if mirror["catalog_sha256"] != manifest["catalog_sha256"]:
            raise ValueError("Mirror belongs to a different source catalog")
        self.mirror = copy.deepcopy(mirror)
        super().__init__(prefix, manifest, tokenizer, rank, world)
        shards = {
            shard["uri"]: shard
            for source in self.catalog.values()
            for shard in source["shards"]
        }
        objects = self.mirror["objects"]
        if set(objects) != set(shards):
            raise ValueError("Mirror must cover exactly the complete source catalog")
        destinations = set()
        for uri, shard in shards.items():
            item = objects[uri]
            if not item["uri"].startswith(self.mirror["root"].rstrip("/") + "/"):
                raise ValueError("Mirror object is outside its declared storage root")
            if item["uri"] in destinations:
                raise ValueError("Distinct source shards share a mirror destination")
            destinations.add(item["uri"])
            if item["source_etag"] != shard["etag"]:
                raise ValueError("Mirror was verified against a different source object")
            if item["metadata"].get("size") != shard["size"]:
                raise ValueError("Mirror size differs from the source catalog")
            if not any(
                value is not None
                for key, value in item["metadata"].items()
                if key != "size"
            ):
                raise ValueError("Mirror requires a destination object version")
            checksum = item["sha256"]
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("Mirror requires a verified content SHA256")

    def shard_location(self, shard: dict) -> tuple[str, dict[str, Any]]:
        """Use destination metadata while the base stream hashes the original URI."""
        item = self.mirror["objects"][shard["uri"]]
        return item["uri"], item["metadata"]
