# cmake -DIN=task_graph.json -DOUT=task_graph_blob.inc -P task_graph_blob.cmake
# The task graph as a NUL-terminated byte array (fleet_mpk_task_graph), included by the provider's kernel.hip.
file(READ ${IN} _hex HEX)
string(LENGTH "${_hex}" _n)
math(EXPR _bytes "${_n} / 2 + 1")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _arr "${_hex}")
file(WRITE ${OUT} "static const unsigned char fleet_mpk_task_graph[${_bytes}] = {${_arr}0x00};\n")

