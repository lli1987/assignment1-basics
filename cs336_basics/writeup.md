Problem 1:

a) What Unicode character does chr(0) return?
'\x00'

b) How does this character’s string representation (__repr__()) differ from its printed representation?
"'\\x00'"

c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:

this character is null

Problem 2:
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

If the text is mostly ascii characters, utf-8 is more efficient 

b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.

For any non-ascii bytes such as 'hello! こんにちは!', it returns UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data error, since the single byte cannot map to any valid unicode character.