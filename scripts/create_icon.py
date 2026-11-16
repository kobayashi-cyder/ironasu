"""Deterministic multi-size ironasu Windows icon; no image dependencies."""
from __future__ import annotations

from pathlib import Path
import struct
import sys
import zlib


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)


def _distance(x, y, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((x-a[0])*vx + (y-a[1])*vy)/(vx*vx+vy*vy)))
    return ((x-(a[0]+t*vx))**2+(y-(a[1]+t*vy))**2)**0.5


def png(size: int) -> bytes:
    corners = ((.5,.15),(.85,.5),(.5,.85),(.15,.5))
    edges = tuple((corners[i],corners[(i+1)%4]) for i in range(4))
    rows = []
    for j in range(size):
        row = bytearray([0])
        for i in range(size):
            x,y = (i+.5)/size,(j+.5)/size
            dx,dy=max(0,abs(x-.5)-.37),max(0,abs(y-.5)-.37)
            if dx*dx+dy*dy > .01:
                row += b"\x00\x00\x00\x00"
                continue
            rgba = (21,52,56,255)
            if min(_distance(x,y,a,b) for a,b in edges) < .03:
                rgba = (137,231,189,255)
            if max(abs(x-.5),abs(y-.5)) < .042:
                rgba = (230,255,247,255)
            row += bytes(rgba)
        rows.append(bytes(row))
    head = struct.pack(">IIBBBBB",size,size,8,6,0,0,0)
    return b"\x89PNG\r\n\x1a\n"+_chunk(b"IHDR",head)+_chunk(b"IDAT",zlib.compress(b"".join(rows),9))+_chunk(b"IEND",b"")


def create_icon(target: Path) -> None:
    sizes = (16,24,32,48,64,128,256)
    images = [png(n) for n in sizes]
    offset = 6+16*len(images)
    data = bytearray(struct.pack("<HHH",0,1,len(images)))
    for n,image in zip(sizes,images):
        data += struct.pack("<BBBBHHII",n if n<256 else 0,n if n<256 else 0,0,0,1,32,len(image),offset)
        offset += len(image)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(bytes(data)+b"".join(images))


if __name__ == "__main__":
    create_icon(Path(sys.argv[1] if len(sys.argv)>1 else "dist/ironasu.ico"))
