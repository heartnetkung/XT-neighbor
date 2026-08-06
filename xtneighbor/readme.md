# Deprecation warning

The non-streaming version of XTNeighbor is only provided for proof of concept. It is not meant for production use. Please use SymScan or XTNeighbor-streaming instead.

Note that this implementation silently truncates strings of length >18, and does not check correctness on the full length string. This is unlike XTNeighbor-streaming which also uses prefix truncation but filters any false positives.
