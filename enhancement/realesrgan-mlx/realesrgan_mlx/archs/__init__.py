"""Architecture modules — isomorphic to basicsr/archs/."""

from .rrdbnet_arch import RRDB, RRDBNet, ResidualDenseBlock
from .srvgg_arch import SRVGGNetCompact

__all__ = ["RRDBNet", "RRDB", "ResidualDenseBlock", "SRVGGNetCompact"]
