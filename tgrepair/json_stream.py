"""Bounded-memory iteration over an array in a top-level JSON object."""
import json


class _Reader:
    def __init__(self,stream,chunk_size):
        self.stream=stream;self.chunk_size=chunk_size
        self.buffer='';self.offset=0;self.eof=False
        self.decoder=json.JSONDecoder()

    def more(self):
        self.buffer=self.buffer[self.offset:];self.offset=0
        chunk=self.stream.read(self.chunk_size)
        self.buffer+=chunk
        if not chunk:self.eof=True

    def peek(self):
        while True:
            while self.offset<len(self.buffer) and self.buffer[self.offset].isspace():self.offset+=1
            if self.offset<len(self.buffer):return self.buffer[self.offset]
            if self.eof:raise ValueError('Unexpected end of JSON input')
            self.more()

    def take(self,expected):
        if self.peek()!=expected:raise ValueError(f'Expected JSON delimiter {expected!r}')
        self.offset+=1

    def value(self):
        self.peek()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer,self.offset)
                # A numeric token can end at a chunk boundary while more digits
                # remain. Obtain a delimiter before accepting any such value.
                if end==len(self.buffer) and not self.eof:
                    self.more();continue
                self.offset=end
                return value
            except json.JSONDecodeError as error:
                if self.eof:raise ValueError('Invalid or incomplete JSON value') from error
                self.more()


def iter_top_level_array(stream,name,chunk_size=1024*1024):
    """Skip other root arrays item by item; never match a nested field name."""
    reader=_Reader(stream,chunk_size);reader.take('{')
    while reader.peek()!='}':
        key=reader.value()
        if not isinstance(key,str):raise ValueError('JSON object key is not a string')
        reader.take(':')
        if reader.peek()=='[':
            reader.take('[')
            while reader.peek()!=']':
                item=reader.value()
                if key==name:yield item
                if reader.peek()==']':break
                reader.take(',')
            reader.take(']')
            if key==name:return
        else:
            reader.value()
        if reader.peek()=='}':break
        reader.take(',')
    raise ValueError(f'Top-level array {name!r} not found')
