#pragma once
void _local_scatter_launch(half* input, half* output, int hidden_size, int* pos, int pos_len);
void _local_gather_launch(half* input, half* output, int hidden_size, int* pos, int pos_len);